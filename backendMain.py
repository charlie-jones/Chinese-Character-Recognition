from flask import Flask
from flask import render_template
from flask import request
from NeutralNetwork import NeuralNetwork, NeuralLayer, NeuralNode
from Filter3x3 import Filter3x3
from Filter3x3 import getCharacter
import pinyin
import pinyin.cedict
from numpy import reshape, array

app = Flask(__name__)
filter = Filter3x3()
filter.readWeights()
filter.readBiases()
filter.saveFilters()

@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == "POST":
		cDta = request.form["data"] # now cDta is 2d array
		
		# pass through neural network to get label

		# PREVIOUS CODE:

		# nn = NeuralNetwork(16384, 6825, 5, 1)
		# output = nn.feedForward(nn.readImageData(name), 0); 
		# output = nn.feedForward(nn.readImageData(cDta), 0); 
		# character = nn.getCharacter(output)
		cDta = reshape(cDta.split(','), (128,128))
		cDta = array(cDta, dtype='int')
		character = getCharacter(cDta, filter)
		character+=1

		print(character)
		print(pinyin.get(character))
		print(pinyin.cedict.translate_word(character))
		# print(pinyin.get('你 好'))
		# print(pinyin.cedict.translate_word('你好'))
		
		# from label get the Chinese character
		# translate character into english and pinyin
		# return the english and pinyin (which should put as a string on the page)
		
		# this prints the character, pinyin, and english on the page (we can change the formatting of this so it looks better)
		rtn = "character: " + character + "<br>pinyin: " + pinyin.get(character) + "<br>english: " + str(pinyin.cedict.translate_word(character))
		return rtn
		# return "blahhhh
		
	return render_template("main.html")
	

#@app.route('/get_names', methods=['POST'])
#def get_names():
   #if request.method == 'POST':
   	 #name = request.form["data"]
   	 
	   

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

