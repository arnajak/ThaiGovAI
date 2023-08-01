import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

@st.cache_data
def load_model():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained("Arnajak/mt5_base-thai_government_parapharse")
  model = AutoModelForSeq2SeqLM.from_pretrained("Arnajak/mt5_base-thai_government_parapharse").to(device)
  return model, tokenizer

def post_process(informal_text,pred,sim_model=None):

    informal_to_formal = {
    'ใคร': 'ผู้ใด',
    'ที่ไหน': 'ที่ใด',
    'แบบไหน': 'แบบใด',
    'อะไร': 'สิ่งใด',
    'ได้ไหม': 'ได้หรือไม่',
    'เมื่อไหร่': 'เมื่อใด',
    'อย่างไร': 'เช่นใด',
    'ทำไม': 'เพราะอะไร',
    'เดี๋ยวนี้': 'ขณะนี้บัดนี้',
    'ต้องการ': 'ประสงค์',
    'มีความ': 'ประสงค์',
    'ช่วย': 'อนุเคราะห์',
    'ไม่ใช่': 'มิใช่',
    'ไม่ดี': 'มิชอบ',
    'ไม่ได้': 'มิได้',
    'ในเรื่องนี้': 'ในการนี้ในกรณีนี้',
    'เรื่องนั้น': 'กรณีดังกล่าว',
    'เหมือนกัน': 'เช่นเดียวกัน',
    'ขอเชิญมา': 'ขอเชิญไป',
    'ยังไม่ได้ทำเลย': 'ยังไม่ได้ดำเนินการแต่อย่างใด',
    'เสร็จแล้ว': 'เรียบร้อยแล้ว',
    'ขอเตือนว่า': 'ขอเรียนให้ทราบว่า',
    'ปัญญาทึบ': 'ขาดความรู้ความเข้าใจ',
    'โง่': 'ขาดความรู้ความเข้าใจ',
    'ใช้ไม่ได้': 'ยังบกพร่อง',
    'เลว': 'ยังบกพร่อง',
    'ขอให้ดำเนินการ': 'โปรดพิจารณาดำเนินการ',
    'โปรดอนุมัติ': 'โปรดพิจารณาอนุมัติ',
    'ใช้จ่าย' : 'เบิกจ่าย',
    'ข้าพเจ้า' : "กระผม",
    "ซื้อของ" : "ซื้ออุปกรณ์"
    }

    # clear hallucination at the beginning
    begin_pred = pred[:9]
    result = re.search(r"[\d+๐-๙]+",pred)
    if result != None and re.search(r"[\d๐-๙]",informal_text) == None:
        pred = pred[result.end():]


    # informal2formal
    pattern = '|'.join(map(re.escape, informal_to_formal.keys()))
    pred = re.sub(pattern, lambda m: informal_to_formal[m.group()], pred)

    if len(pred) < 3 : 
       pred = "ยังไม่สามารถทำได้ในขณะนี้"

    return pred.strip()

def prediction(user_input,model,tokenizer,sim_model=None):
    text = "paraphrase: "+ user_input
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = tokenizer(text , return_tensors='pt').to(device)
    output_tokens = model.generate(**batch,max_length=768,top_k=50, top_p=0.95, temperature=1)
    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    output = post_process(user_input,output,sim_model)
    st.session_state.output = output


def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Thai2GovAI V.0.1</h1>", unsafe_allow_html=True)
    model, tokenizer = load_model()
    # sim_model = load_sim()
    col1, col2, col3= st.columns([8,1,8])
    st.session_state.input = ""
    st.session_state.output = ""
    with col1:
      st.markdown("<h3 style='text-align: center; color: white;'>Informal style</h3>", unsafe_allow_html=True)
      option = st.selectbox('Select example',options=("Select example",'Example 1', 'Example 2', 'Example 3'))
      if option == "Example 1":
        st.session_state.input = "เพื่อให้เราแก้ไขปัญหาความไม่สงบได้อย่างสงบสุข โดยไม่เสียชีวิตหรือทรัพย์สินของคนในทีม จึงขอเชิญคุณมาร่วมประชุมในวันที่ 20 พฤษภาคม พุทธศักราช 2557 เวลา 14:00 "
      elif option == "Example 2":
        st.session_state.input = "ห้ามขายสินค้าที่ผิดกฎหมาย"
      elif option == "Example 3":
        st.session_state.input = "ข้อ 5 ติดตามค่าใช้จ่ายด้านการเรียน"
      user_input = st.text_area("",st.session_state.input,height =300,label_visibility="collapsed",key=1)

    with col2:
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      button = st.button('➜')
      if button:
        prediction(user_input,model,tokenizer) 

    with col3:
      st.markdown("<h3 style='text-align: center; color: white;'>Government style</h3>", unsafe_allow_html=True)
      rate = st.selectbox('Please rate my answer!',options=('5', '4', '3','2','1'))
      st.text_area("",st.session_state.output,label_visibility="collapsed", height =300,key=2)


if __name__ == "__main__":
    main()
