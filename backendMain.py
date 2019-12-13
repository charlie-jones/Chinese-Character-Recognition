from flask import Flask
from flask import render_template
import pinyin
import pinyin.cedict

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
	return render_template('main.html')

if __name__ == '__main__':
	app.run(debug=True)

"""Translates text into the target language.

Target must be an ISO 639-1 language code.
See https://g.co/cloud/translate/v2/translate-reference#supported_languages
"""
'''from google.cloud import translate_v2 as translate
translate_client = translate.Client()

if isinstance(text, six.binary_type):
    text = text.decode('utf-8')
 
#Text can also be a sequence of strings, in which case this method
#will return a sequence of results for each text.
result = translate_client.translate(
    text, target_language=target)

print(u'Text: {}'.format(result['input']))
print(u'Translation: {}'.format(result['translatedText']))
print(u'Detected source language: {}'.format(
    result['detectedSourceLanguage']))'''




'''
print pinyin.get('你 好')
print pinyin.get('你好', format="strip", delimiter=" ")
print pinyin.get('你好', format="numerical")
print pinyin.get_initial('你好')
'''

"""
pinyin.cedict.translate_word('你')
pinyin.cedict.translate_word('你好')
list(pinyin.cedict.all_phrase_translations('你好'))
"""

