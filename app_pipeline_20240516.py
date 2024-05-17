import os
from openai import OpenAI
import httpx

import gradio as gr
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor
from openai import AsyncOpenAI, OpenAI
from huggingface_hub import login

# 環境設定
# 若沒設，H100會擋
#os.environ["no_proxy"] = "localhost, 127.0.0.1,::1"

# hugginface login
#login()

async def get_completion(client, model, content):
  for _ in range(3):
      try:
          resp = await client.chat.completions.create(
              model=model,
              messages=content,
              temperature=0.2,
              timeout=180,
              seed=42
          )
          return parse_score(resp.choices[0].message.content)
      except Exception as e:
          # time.sleep(1)
          print(e)
          await asyncio.sleep(10)
  print('ignore', content)
  return {"score": -1, "judge_response": "error"}

async def get_completion_list(model, content_list, batch_size=20):
  client = AsyncOpenAI()
  result = await tqdm_async.gather(*[get_completion(client, model, content) for content in content_list], )

  return result

def create_instruction(template: dict[str, str],
                       question: str,
                       responses: list[str],
                       ground_truth: list[str] = None):
    user_dct = {'question': question.strip()}
    for i, resp in enumerate(responses):
        user_dct[f'answer_{i+1}'] = resp.replace('\u200b', '')[:2000]

    # ground_truth
    user_dct['ground_truth'] = ground_truth
    user_context = template['user'].format(**user_dct)

    return [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": user_context}
    ]

def parse_score(review):
    try:
        score = review.split("\n")[-1].split(":")[-1]
        if '/10' in score:
            score = score.replace('/10', '')
        return {
            "score": eval(score),
            "judge_response": review,
        }
    except Exception as e:
        print(score)
        return {
            "score": 0,
            "judge_response": review,
        }

def compute_score(result, judge_resps):
  for prompt, resp in zip(result, judge_resps):
        prompt['judge_response'] = resp['judge_response']

        prompt['score'] = resp['score']

  # overall score
  overall_scores = 0
  result = [r for r in result if r['score'] != -1]
  for r in result:
      overall_scores += r['score']

  # save result
  with open('./hahah.json', 'w') as f:
    json.dump({"overall": {"score": overall_scores,
                "avg_score": overall_scores / len(result)},
          "result": result},
          f, indent=2, ensure_ascii=False)
  print(overall_scores / len(result))
  print(result[0]['judge_response'].split("\n\n")[0])
  return overall_scores / len(result), result[0]['judge_response'].split("\n\n")[0]
  

def evaluation(gen_dct):
  # 讀取摘要評估prompt
  template_path = "template_judge/geval_summarization.json"
  with open(template_path) as f:
    template = json.load(f)
  # 標準答案
  gt_dct = {'resp': "事實上，高溪池在基隆市擔任新聞採訪工作48年，在新聞界擁有「老報人」封號，曾任民眾日報、青年日報、大華晚報、正聲電台、台灣日報基隆特派員，還擔任過基隆市外勤記者聯誼會理事長。不只新聞界，高溪池在政壇也有舉足輕重的地位，他曾出過8本《基隆選舉錄》，一路從最基層里長、議員、市長，到國大代表、立委、省長、總統大選都全記錄。",
        'prompt': gen_dct['prompt'],
        'model': "grounf_truth",    
        'qid': 0
  }
  # ground_truth and generation pairs
  resp_dct = {gt_dct['model']: gt_dct,
         gen_dct['model']: gen_dct
  } 
  result = []
  resp_names = list(resp_dct.keys())
  question = resp_dct[resp_names[0]]['prompt'][0]['content']
  print("question:", question)
  model_resps = [resp_dct[name]['resp'] for name in resp_names]

  # 存放ground_truth and generation result
  result.append({
    "qid": 0,
    "question": gt_dct['prompt'],
    "model_responses": {name: resp_dct[name]['resp'] for name in resp_names},
    "eval_instruction": create_instruction(template, question, model_resps)
  })
  print(result[0]["model_responses"])
  print(result[0]["eval_instruction"])

  eval_instruction = [r.pop('eval_instruction') for r in result]
  judge_model = 'gpt-3.5-turbo'
  req_method = 'async'
  if req_method == 'async':
    judge_resps = asyncio.run(get_completion_list(
        judge_model, eval_instruction), debug=True)

  overall_scores, comments = compute_score(result, judge_resps)
  return overall_scores, comments


def get_model(model_name):
    """
    透過pipeline載入模型
    """
    model_type = {
        'yi-6B-chat': '../../../../../../raid2/model/models--01-ai--Yi-6B-Chat/snapshots/63d431abe178700a8c143a733e65caf6ed5a2614',
        'TAIDE-LX-7B-Chat': 'taide/TAIDE-LX-7B-Chat',
        'TAIDE_llama3_8B-chat': '../../../../../../raid2/model/TAIDE_llama3_8B-chat',
        'Meta-Llama-3-8B-Instruct': '../../../../../../raid2/model/Meta-Llama-3-8B-Instruct',
    }

    model_id = model_type.get(model_name)
    dtype = torch.bfloat16
    # load taide model 
    pipe = pipeline("text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",)

    print('model loading done...')
    return pipe

def generate(prompt, pipe):
    """
    透過輸入文章產生對應輸出
    """
    terminators = [pipe.tokenizer.eos_token_id,
                   pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipe(prompt,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    #do_sample=True,
                    #temperature=0.6,
                    #top_p=0.9
    )
    return outputs[0]["generated_text"][len(prompt):]


def summary(text, model_name):
    """
    產生出摘要
    text : receive input text
    model_name : receive chosen model
    """
    print(model_name)
    # 讀取模型
    #pipe = get_model(model_name)
    print("done...")

    # 中英文回答設定
    
    if model_name == "Meta-Llama-3-8B-Instruct":
      message = f"""<s>[INST]幫我摘要，並以繁體中文回答\n\n{text} [/INST]"""
    else:
      message = f"""<s>[INST]幫我摘要\n\n{text} [/INST]"""
        
    message = [{'role': "user", "content": message}]
    question = {'resp': '', 'prompt': message, 'model': model_name, 'qid': 0} 
    """
    prompt = pipe.tokenizer.apply_chat_template(question['prompt'], 
                            tokenize=False, 
                            add_generation_prompt=True
    )
    # 產生回覆
    resopnse = generate(prompt, pipe)
    """
    resopnse = " 一名網友在母親罹患新冠肺炎後併發症及糖尿病導致健康惡化，並在病床上與母親拍下最後一張合照，表達對母親的深情道別。原PO回憶母親一生的關愛與付出，感嘆孩子長大後往往要面對的是父母健康的惡化。文章充滿感傷，引起網友共鳴，淚崩之餘也更加珍惜與家人相處的時光。\n原PO敘述其母親在去年染疫，之後可能因新冠後遺症及糖尿病問題，心血管與腎臟功能出現問題，醫生提醒家屬要多加看管。之後，母親發生血糖過低昏迷的事件，原PO雖提醒母親若有不適要聯繫，但隨後卻再也沒有醒過來，最終在抵達醫院前不幸過世。"
    # 加入字典
    gen_dct = {'resp': resopnse,
          'prompt': question['prompt'],
          'model': model_name,
          'qid': question['qid']
    }
    print("gen_dct:", gen_dct)
    overall_scores, comments = evaluation(gen_dct)
    overall_scores = f"{overall_scores}/10"

    return gen_dct['resp'], overall_scores, comments

# 產生翻譯
def translation(text, model_name):
    print(model_name)
    return f"這裡是翻譯"

def clear_data():
  return "", "", "", "", ""



if __name__ =="__main__":
  with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# TAIDE BENCH EVALUATION😊")
    # 這裡是摘要...
    with gr.Tab("摘要"):
      gr.Markdown("""## 透過欲使用模型生成摘要後，再透過GPT4進行評估\n
      評估指南：
      - 簡單明瞭：檢查是否簡單明瞭的保留原始文章大致內容，避免陷入不重要的細節。
      - 用詞選擇：檢查使用的詞彙是否符合台灣中文的習慣，且該使用原文時保留原始語言。
      """)
      chosen_model1 = gr.Dropdown(["TAIDE-LX-7B-Chat", "TAIDE_llama3_8B-chat", "Meta-Llama-3-8B-Instruct", "Yi-1.5-6B-Chat", "mistralai/Mistral-7B-Instruct-v0.2"], label="欲使用模型")
      text_input1 = gr.Textbox(lines=3, placeholder="請輸入文章", label="Context")
      with gr.Row():
        text_button1 = gr.Button("Generate")
        clear_button1 = gr.Button("Clear")
      text_output1 = gr.Textbox(lines=3, label="Summary")
      gr.Markdown("""
      分數說明：
      - 1-3分：摘要明顯有誤或回答原文。
      - 4-6分：摘要存在一些明顯的錯誤或遺漏。
      - 7-8分：摘要大致上是正確的，但還有一些小問題。
      - 9-10分：摘要非常精確，幾乎沒有任何問題。
      """)
      score_optput1 = gr.Textbox(label="Score")
      comment_output1 = gr.Textbox(label="評語")

    # 這裡是翻譯...   
    with gr.Tab("英翻中"):
      chosen_model2 = gr.Dropdown(["TAIDE-LX-7B-Chat", "TAIDE_llama3_8B-chat", "Meta-Llama-3-8B-Instruct", "Yi-1.5-6B-Chat", "mistralai/Mistral-7B-Instruct-v0.2"], label="欲使用模型")
      text_input2 = gr.Textbox(label="輸入")
      text_button2 = gr.Button("Generate")
      text_output2 = gr.Textbox(label="輸出")
      #score_optput2 = gr.Textbox(label="Score")
        
    text_button1.click(fn=summary, inputs=[text_input1, chosen_model1], outputs=[text_output1, score_optput1, comment_output1])
    clear_button1.click(fn=clear_data, inputs=[], outputs=[text_input1, chosen_model1, text_output1, score_optput1, comment_output1])
    text_button2.click(translation, inputs=[text_input2, chosen_model2], outputs=[text_output2])

  #demo.launch(ssl_verify=False)
  demo.launch(share=True)


