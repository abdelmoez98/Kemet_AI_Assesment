import pickle
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

with open('lang_det.pickle','rb') as f:
    det_model = pickle.load(f)
with open('vectorizer.pickle','rb') as f:
    vectorizer = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def translate_to_arabic(text):
  encoded = tokenizer(text, return_tensors="pt")
  generated_tokens = model.generate(
      **encoded,forced_bos_token_id=tokenizer.lang_code_to_id["arb_Arab"]
  )
  return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def translate_to_english(text):
  encoded = tokenizer(text, return_tensors="pt")
  generated_tokens = model.generate(
      **encoded,forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
  )
  return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

app = Flask(__name__)

@app.route('/detect_langauge', methods=['POST'])
def detect_language():

    text = request.json['text']

    if not any(c.isalpha() for c in text):
        return  {'Output': 'The string contains no letters'}
    
    vectors = vectorizer.transform([text])
    out = det_model.predict(vectors)[0]    
    result = {'Output': out}

    return jsonify(result)

@app.route('/translate', methods=['POST'])
def verify_address():

    text = request.json['text']

    if not any(c.isalpha() for c in text):
        return  {'Output': 'The string contains no letters'}

    vectors = vectorizer.transform([text])
    out = det_model.predict(vectors)[0]    

    if out == 'English':
       translated_text = translate_to_arabic(text)[0]
       en_text = text
       ar_text = translated_text
    elif out == 'Arabic' :
       translated_text = translate_to_english(text)[0]
       ar_text = text
       en_text = translated_text
    else:
       return  {'Output': "Error in Language Detection"}
    
    result = {'Output': f'AR: {ar_text}\nEN: {en_text}'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)