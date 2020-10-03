#!/usr/bin/env python

#Modifies DBPedia Topic Signatures

from codecs import open as copen
from collections import defaultdict, Counter
from sys import argv
from math import sqrt
from itertools import chain
from datetime import datetime
from subprocess import call
import re

from lda_gensim import stopwords as STOPWORDS;STOPWORDS=STOPWORDS() #maybe a better way of doing that
from dataset_preparation import QueryFinder

#REMOTE_DIR = "http://wifo5-04.informatik.uni-mannheim.de/downloads/datasets/" #3.5m raw topic signatures 
#LOCAL_SIGNATURES_FILE = "topic_signatures_en.tsv"

def ngrams(s, n):
	"""Given a string s, splits it into ngrams"""
	ngrams = []
	s = s.split()
	for i in range(len(s)-n+1):
		ngrams.append(' '.join(s[i:i+n]))
	return ngrams

def stoplist_ngrams(s, n):
	"""Converts a string to ngrams.
	Avoids concatenating words that had a stopword between them."""
	#Process looks like:
	#1) the cat sat on the mat
	#2) _ cat sat _ _ mat
	#3) bigrams: "cat sat"
	#rather than "cat sat", "sat mat"
	
	ngrams = []
	s = s.lower().split()
	for i in range(len(s)-n+1):
		ngram = s[i:i+n]
		add = True
		for x in ngram:
			if x in STOPWORDS:
				add = False
				break
		if add:
			ngrams.append(' '.join(ngram))
	return ngrams

def cosim(bow, category_keywords):
	"""How similar are two bags of words?"""
	
	#generate vector
	vector = defaultdict(lambda: [0, 0])
	for n, repo in enumerate([bow, category_keywords]):
		for kw in repo:
			vector[kw][n] += 1
	
	dot_product = 0
	mag1 = 0
	mag2 = 0
	for x in vector.values():
		dot_product += (x[0] * x[1])
		mag1 += x[0]**2
		mag2 += x[1]**2
	
	denom = (sqrt(mag1) * sqrt(mag2))
	if denom != 0:
		return dot_product / denom
	else:
		return 0

def find_largest_keyword_divisor(category_dictionary):
	"""Finds words which separate categories the most,
	so that we can split them into a binary tree"""
	
	#get the keyword counts
	d = defaultdict(int)
	for category, keywords in category_dictionary.iteritems():
		for k in keywords.iterkeys():
			d[k] += 1
	
	#rank them in terms of how near they are to half the category size
	half_of_articles = len(d) / 2 #returns int
	
	keyword, distance = "", half_of_articles
	
	for kw, category_count in d.iteritems():
		tmp_distance = abs(category_count-half_of_articles)
		if tmp_distance == 0:
			print [kw, 0]
			exit()
		else:
			if tmp_distance < distance:
				keyword = kw
				distance = tmp_distance
	return [keyword, distance]

def split_category_dictionary(category_dictionary, keyword):
	"""Returns two dictionaries, one which has things with the keyword,
	and one without"""
	with_keyword = {}
	without_keyword = {}
	
	for category, keywords in category_dictionary.iteritems():
		if keyword in keywords:
			with_keyword[category] = keywords
		else:
			without_keyword[category] = keywords
	
	return {'with': with_keyword, 'without': without_keyword}

def get_binary_node(tree, path):
	"""Gets the value from a particular key path since we can't use dot notation"""
	return reduce(lambda branch, key: branch[key], path, tree)

def set_binary_node(tree, path, value):
	"""Sets a binary node with a value"""
	get_binary_node(tree, path[:-1])[path[-1]] = value

def build_divisor_tree(category_dictionary, max_depth):
	"""Builds a tree from keyword divisors at a given depth"""
	
	binary_tree = {}
	
	queue = [[category_dictionary, 0, []]] #dictionary, current_depth, path
	
	while queue[0][1] <= max_depth:
		
		#get next item from the queue
		process_dictionary = queue.pop(0)
		dictionary = process_dictionary[0]
		depth = process_dictionary[1]
		path = process_dictionary[2]
		
		#crunch
		kw = find_largest_keyword_divisor(category_dictionary)
		split_dicts = split_category_dictionary(category_dictionary)
		
		#add to tree
		if path:
			set_binary_node(binary_tree, path+[kw], [split_dicts['with'], split_dicts['without']])
		else:
			binary_tree[kw] = [split_dicts['with'], split_dicts['without']]
		
		
		#add more to the queue
		queue.append(split_dicts['with', depth+1, []])
		queue.append(split_dicts['without', depth+1, []])

def make_ontology_file():
	wikipedia_page_keywords = {}		#3) "Down_to_Earth_%28Justin_Bieber_song%29": ['one', 'girl']

	with copen("topic_signatures_en.tsv", encoding='utf8') as raw:
		for n, line in enumerate(raw):
			
			line = line[:-1].split('\t') #remove the newline character and separate title from rest
			
			wiki_article_title = line[0] #useful
			
			rest = line[1].split('"')
			page_text_salient_keywords = [x for x in rest[-1].split() if x not in STOPWORDS] #useful
			
			wikipedia_page_keywords[wiki_article_title] = page_text_salient_keywords
			
			if n % 100000 == 0:
				print "Processed {0}% of the pages".format((n/3500000.0)*100)
		print "Total: {0} articles".format(len(wikipedia_page_keywords))
	
	with copen("article_category_matrix.tsv", encoding='utf8') as f:
		#has 144k categories, 97k without numbers
		
		article_phrase_matrix = defaultdict(lambda: defaultdict(int))
		
		for n, line in enumerate(f):
			line = line.split("\t")
			category = line[0]
			if not re.match('.*[0-9].*', category): #as long as the category doesn't have a number in it
				articles = line[1:]
				for article in articles:
					if article in wikipedia_page_keywords:
						for phrase in wikipedia_page_keywords[article]:
							article_phrase_matrix[category][phrase] += 1
			if n % 10000 == 0:
				print "Processed {0}".format(n)
	
	#now export in the form:
	#category \t phrase \t count \t phrase \t count
	with copen('payload.lwca', 'w', encoding='utf8') as f:
		for category, words in article_phrase_matrix.iteritems():
			phrases = []
			for phrase, count in sorted(words.items(), key=lambda x: x[1], reverse=True):
				phrases.append(u"{0}\t{1}".format(phrase, count))
			
			f.write(u"{0}\t{1}\n".format(category, '\t'.join(phrases)))

class Ontology:
	def __init__(self, payload_location):
		"""Imports payload"""
		self.category_keywords = {}
		with copen(payload_location, encoding='utf8') as f:
			for line in f:
				line = line.split("\t")
				category = line[0]
				keywords = {}
				for x in range(0, len(line[1:]), 2):
					keywords[line[x+1]] = int(line[x+2])
				self.category_keywords[category] = keywords
		#pruning
		#import words file
		#words = []
		#with copen('/usr/share/dict/words', encoding='utf8') as f:
		#	for line in f:
		#		line = line[:-1]
		#		words.append(line.lower())
		#words = set(words)
		to_save = {}
		for key, value in self.category_keywords.iteritems():
			if len(value) >= 25:
				to_save[key] = value
		self.category_keywords = to_save
		
		#save category keywords to file
		#with copen('pruned_category_keywords.txt', 'w', encoding='utf8') as f:
		#	for k in self.category_keywords.iterkeys():
		#		f.write(k + u"\n")
		
		
		print "Classification algorithm has {0} wikipedia categories to pick from".format(len(self.category_keywords))
		
		d = defaultdict(list)
		with copen("freebase.txt", encoding='utf8') as f:
			for line in f:
				if len(line) > 5:
					line = line[:-1]
					line = line.split("\t")
					category = line[0]
					terms = line[1:]
					for t in terms:
						t = t.lower().split()
						for x in t:
							d[category].append(x)
		self.freebase = {}
		for k,v in d.iteritems():
			self.freebase[k] = Counter(v)
		
		print "Classification algorithm has {0} freebase categories to pick from".format(len(self.freebase))
		
		#now try with IAB
		
		self.iab_bigrams = defaultdict(list)
		self.iab_keywords = defaultdict(list)
		
		with copen("iab_wiki_bigrams.tsv", encoding='utf8') as f:
			for line in f:
				if line != "":
					line = line[:-1].split("\t")
					category = line[0]
					for x in range(0, len(line[1:]), 2):
						for i in range(int(line[x+2])):
							self.iab_bigrams[category].append(line[x+1])
		
		with copen("iab_wiki_keywords.tsv", encoding='utf8') as f:
			for line in f:
				if line != "":
					line = line[:-1].split("\t")
					category = line[0]
					for x in range(0, len(line[1:]), 2):
						for i in range(int(line[x+2])):
							self.iab_keywords[category].append(line[x+1])
		
		self.query_finder = QueryFinder()
	
	def classify_webpage(self, url, title): #might change this to *kwargs so users can just throw anything at it 
		"""Attempts to classify a given webpage. Returns 10 categories"""
		
		#basically does a union of the ngrams in the data with each of the 97 categories
		#how fast?
		title = re.findall("[a-z]+", title.lower())
		
		matches = []
		
		for category, words in self.category_keywords.iteritems():
			kws = [item for sublist in [[x[0]]*x[1] for x in words.iteritems()] for item in sublist]
			matches.append([category, cosim(title, kws)])
		
		matches = sorted(matches, key=lambda x: x[1], reverse=True)[:10]
		return matches
	
	def build_inverse_index(self):
		"""Could be very inefficient but this builds an inverse index"""
		article_to_id = {}
		self.id_to_article = {}
		self.inverse_index = defaultdict(set)
		
		for category, keywords in self.category_keywords.iteritems():
			index = len(article_to_id)
			article_to_id[category] = index
			self.id_to_article[index] = category
			for k in keywords.iterkeys():
				self.inverse_index[k].update([index])
		
		print "inverse index is of size: {0} and id_to_article is {1}".format(len(self.inverse_index), len(self.id_to_article))
	
	def classify_webpage_with_ii(self, title):
		"""Attempts to classify a given webpage using the inverse index.
		Returns the top 10 categories"""
		
		print "-----------------------------------"
		print "Debugging ii webpage classification"
		
		title = re.findall("[a-z]{3,}", title.lower())
		
		print "title: {0}".format(title)
		
		matches = []
		
		#can't do the actual intersection since cosim doesn't need exact matches.
		#just needs articles with at least 1 of the matches
		
		articles = set([]) # a set of articles worth looking at, auto-deduped
		
		word_articles = {} #diagnostics
		
		for keyword in title:
			articles.update(self.inverse_index[keyword])
			
			if keyword not in word_articles:
				word_articles[keyword] = self.inverse_index[keyword]
		
		print "diagnostics as follows:"
		for word in sorted(word_articles.keys()):
			print word, len(word_articles[word])
		print "total of {0} words".format(len(word_articles))
		print "which have a total of {0} articles".format(sum([len(x) for x in word_articles.itervalues()]))
		uniques = []
		for x in word_articles.itervalues():
			uniques += x
		uniques = set(uniques)
		print "{0} of which are unique".format(uniques)
		
		
		
		print "total number of articles to look at: {0}".format(len(articles))
		
		scores = []
		for article_number in articles:
			category = self.id_to_article[article_number]
			words = self.category_keywords[category]
			kws = [item for sublist in [[x[0]]*x[1] for x in words.iteritems()] for item in sublist]
			matches.append([category, cosim(title, kws)])
		
		matches = sorted(matches, key=lambda x: x[1], reverse=True)[:10]
		return matches
	
	def classify_webpage_with_freebase(self, text):
		"""Classifies articles using 2 level freebase categorization system"""
		
		text = re.findall("[a-z]+", text.lower())
		scores = []
		
		for category, keywords in self.freebase.iteritems():
			scores.append([category, cosim(keywords, text)])
		
		scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
		return scores
	
	def classify_webpage_with_iab_keywords(self, text):
		"""Classifies sessions using the IAB keywords dataset"""
		
		text = re.findall("[a-z]+", text.lower())
		scores = []
		
		for category, keywords in self.iab_keywords.iteritems():
			scores.append([category, cosim(keywords, text)])
		
		scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
		return scores
	
	def classify_webpage_with_iab_bigrams(self, text):
		
		text = ngrams(text.lower(), 2)
		scores = []
		
		for category, bigrams in self.iab_bigrams.iteritems():
			scores.append([category, cosim(bigrams, text)])
		
		scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
		return scores

def classify_session_queries(cls_type='freebase', kw_type='kws'):
	"""Classifies user sessions"""
	#have to think of a good diagnostic output
	#need: sample domains, urls, keywords, and then classifications
	#could output a webpage?
	html_start = """
	<html><head><meta charset="UTF-8"> <style>body {font-family: arial;} td, th {border:0.5px solid black;} table {border-collapse: collapse;}</style></head><body><h1>Session Classifications - LWCA v0.2</h1><table><tr><th>#</th><th>Page views</th><th style="width: 50%;">Top Title Keywords</th><th>Classification</th></tr>"""
	html_end = """</table></body></html>"""
	row_template = 	"<tr><td>{{NUMBER}}</td><td>{{PAGEVIEWS}}</td><td>{{KEYWORDS}}</td><td>{{CLASSIFICATION}}</td></tr>\n"
	
	#now have to calculate each session
	from dataset_preparation import session_query_generator
	
	print "Getting user data..."

	sessions = session_query_generator()
	
	print "Importing payload...",
	o = Ontology('payload.lwca')
	print "building index...."
	o.build_inverse_index()
	print "done."
	
	classifications = []
	
	print "Iterating through sessions..."
	for session in sessions:
		
		#if len(classifications) > 30:
		#	break
		
		print "Recieved session with {0} queries".format(len(session))
		
		if len(session) < 5:
			print "Skipped as too short"
			continue
		
		results = {}
		
		results['number'] = len(classifications)
		results['pageviews'] = "n/a"
		
		session_bow = " ".join(session) #session bag of words
		
		if kw_type == "bigrams":
			keywords = ngrams(session_bow, 2)
		else:
			keywords = re.findall("[a-zA-Z]+", session_bow)
		
		formatted_kws = Counter(keywords).most_common(20)
		results['keywords'] = "{0} total, e.g.:\n{1}".format(len(keywords), formatted_kws)
		
		if cls_type == "iab_keywords":
			cls = o.classify_webpage_with_iab_keywords(session_bow)
		elif cls_type == "iab_bigrams":
			cls = o.classify_webpage_with_iab_bigrams(session_bow)
		else:
			cls = o.classify_webpage_with_freebase(session_bow)
		
		results['classification'] = u"<br>".join([u"{0}-{1}".format(x[0], round(x[1],2)) for x in cls if x[1] > 0])
		
		classifications.append(results)
	
	print "Classified {0} sessions".format(len(classifications))
	print "Formatting and saving to sessions.html"
	with copen("sessions.html",'w', encoding='utf8') as f:
		f.write(html_start)
		for x in classifications:
			tmp_row = row_template
			for key, value in x.iteritems():
				tmp_row = tmp_row.replace("{{"+key.upper()+"}}", unicode(value))
			f.write(tmp_row)
		f.write(html_end)
	
	print "Done"

def classify_sessions(cls_type="freebase", kw_type='kws'):
	"""Classifies user sessions"""
	#have to think of a good diagnostic output
	#need: sample domains, urls, keywords, and then classifications
	#could output a webpage?
	html_start = """
	<html>
		<head>
		<meta charset="UTF-8"> 
			<style>
				body {font-family: arial;}
				td, th {border:0.5px solid black;}
				table {border-collapse: collapse;}
			</style>
		</head>
		<body>
			<h1>Session Classifications - LWCA v0.2</h1>
			<table>
				<tr>
					<th>#</th><th>Page views</th><th style="width: 50%;">Top Title Keywords</th><th>Classification</th>
				</tr>
	"""
	html_end = """
			</table>
		</body>
	</html>
	"""
	row_template = 	"<tr><td>{{NUMBER}}</td><td>{{PAGEVIEWS}}</td><td>{{KEYWORDS}}</td><td>{{CLASSIFICATION}}</td></tr>\n"
	
	#now have to calculate each session
	from persistent import remove_persistent_title_components_across_sessions
	from dataset_preparation import sessionized_visit_group_generator
	from dataset_preparation import find_repositories_on_computer
	
	print "Getting user data..."

	dbloc = find_repositories_on_computer()
	dbloc = [x for x in dbloc if "Test" in x][0]
	sessions = sessionized_visit_group_generator(dbloc)

	sessions = remove_persistent_title_components_across_sessions(sessions)
	
	print "Importing payload...",
	o = Ontology('payload.lwca')
	print "building index...."
	o.build_inverse_index()
	print "done."
	
	classifications = []
	
	print "Iterating through sessions..."
	for session in sessions:
		
		print "Recieved session of length {0}".format(len(session))
		
		results = {}
		results["number"] = "error"
		results["pageviews"] = "error"
		results["keywords"] = "error"
		results["classification"] = "error"
		results['number'] = len(classifications)

		pageviews = len(session)
		results['pageviews'] = pageviews
		session_bow = " ".join([x[1] for x in session]) #session bag of words
		
		if kw_type == "bigrams":
			keywords = ngrams(session_bow, 2)
		else:
			keywords = re.findall("[a-zA-Z]+", session_bow)
		
		formatted_kws = Counter(keywords).most_common(20)
		results['keywords'] = "{0} total, e.g.:\n{1}".format(len(keywords), formatted_kws)
		
		if cls_type == "iab_keywords":
			cls = o.classify_webpage_with_iab_keywords(session_bow)
		elif cls_type == "iab_bigrams":
			cls = o.classify_webpage_with_iab_bigrams(session_bow)
		else:
			cls = o.classify_webpage_with_freebase(session_bow)
		
		results['classification'] = u"<br>".join([u"{0}-{1}".format(x[0], round(x[1],2)) for x in cls if x[1] > 0])
		
		classifications.append(results)
	
	print "Classified {0} sessions".format(len(classifications))
	print "Formatting and saving to sessions.html"
	with copen("sessions.html",'w', encoding='utf8') as f:
		f.write(html_start)
		for x in classifications:
			
			tmp_row = row_template
			for key, value in x.iteritems():
			
				matcher = "{{"+key.upper()+"}}"
			
				tmp_row = tmp_row.replace("{{"+key.upper()+"}}", unicode(value))
			
			
			f.write(tmp_row)
		f.write(html_end)
	
	print "Done"

from nltk.corpus import wordnet as wn

def nounerize(thing):
	"""Tries to turn non-nouns into nouns"""
	drf = defaultdict(int) #derivationally related forms
	
	ss = wn.synsets(thing)
	for thing in ss:
		if thing.pos != "n":
			lemmas = thing.lemmas
			for lemma in lemmas:
				for form in lemma.derivationally_related_forms():
					drf[form.name] += 1 
	
	return drf if len(drf) > 0 else None

class PostProcessor:
	def __init__(self):
		"""Loads in Ed and Olivier's domainRules.json file, now converted to a big (7k+ entry) dict object"""
		#import domainRules.json
		from domain_rules import domain_rules
		from tldextract.tldextract import extract
		self.extract = extract
		from nltk.stem.porter import PorterStemmer as PorterStemmer
		self.domain_rules = domain_rules
		#create stemmer
		self.Stemmer = PorterStemmer()
		
	def rerank(self, url, text, results):
		"""Processes classified results"""
		
		#check if the domain exists in domainrules
		domain = self.extract(url)
		domain = domain.domain + "." + domain.suffix
		
		print "Extracted domain: {0}".format(domain)
		
		if domain in self.domain_rules:
			print "found domain"
			if "__ANY" in self.domain_rules[domain]:
				categories = self.domain_rules[domain]['__ANY']
				for cat in categories:
					#stem it
					matchers = [self.Stemmer.stem(cat)]
					if "-" in matchers[0]:
						matchers.append(matchers[0].replace("-", "_"))
					for matcher in matchers:
						for x in range(len(results)):
							print "comparing {0} to {1}".format(matcher, results[x][0])
							if matcher.lower() in results[x][0].lower():
								print "{0} with score {1} contains {2}".format(results[x][0], results[x][1], matcher)
								results[x][1]  = results[x][1] + 1
								print "score is now {0}".format(results[x][1])
		else:
			print "augmenting common words"
			#check for common words
			words = defaultdict(int)
			for result in results:
				tokens = re.findall("[a-z]+", result[0].lower())
				for token in tokens:
					words[token] += 1
			
			#remove single entries
			for k,v in words.iteritems():
				if v > 1:
					for x in range(len(results)):
						matchers = [self.Stemmer.stem(k)]
						if "-" in matchers[0]:
							matchers.append(matchers[0].replace("-", "_"))
						for matcher in matchers:
							if matcher.lower() in results[x][0].lower():
								print "{0} with score {1} contains {2} which has score {3}".format(results[x][0], results[x][1], matcher, v)
								results[x][1] = results[x][1] + v
								print "score is now {0}".format(results[x][1])
		
		return sorted(results, key=lambda x:x[1], reverse=True)

if __name__ == '__main__':
	print "=========================="
	print "++ mruttley - LWCA Demo v0.4 ++"
	print "=========================="
	print """
	1. Classify all sessions using titles
	2. Classify all sessions using queries
	3. Classify sample titles one-by-one
	4. Classify all titles using IAB keywords
	5. Classify all titles+queries using IAB keywords
	6. Classify all titles using IAB bigrams
	7. Classify all titles+queries using IAB bigrams
	8. Classify text one-by-one with ii and result post-processing"""
	
	if len(argv) > 1:
		decision = argv[1]
	else:
		decision = raw_input("Choice: ")
		
	if "1" in decision:
		#output all sessions to the html file
		classify_sessions()
		call(["open", "sessions.html"])
	elif "2" in decision:
		#output to same file, just this time we're using session queries rather than titles
		classify_session_queries()
		call(["open", "sessions.html"])
	elif "4" in decision: #iab keywords + titles
		classify_sessions(cls_type="iab_keywords")
		call(["open", "sessions.html"])
	elif "5" in decision: #iab keywords + titles + queries
		classify_session_queries(cls_type="iab_keywords")
		call(["open", "sessions.html"])
	elif "6" in decision: #iab bigrams + titles 
		classify_sessions(cls_type="iab_bigrams", kw_type='bigrams')
		call(["open", "sessions.html"])
	elif "7" in decision: #iab bigrams + titles + queries
		classify_session_queries(cls_type="iab_bigrams", kw_type='bigrams')
		call(["open", "sessions.html"])
	elif "8" in decision:
		print "Importing payload...",
		o = Ontology('payload.lwca')
		print "building index...."
		o.build_inverse_index()
		print "done."
		p = PostProcessor()
		while True:
			url = raw_input("URL: ")
			text = raw_input("Title and/or Text: ")
			start = datetime.now()
			classification = o.classify_webpage_with_ii(text)
			end = datetime.now()
			delta = end-start
			classification = p.rerank(url, text, classification)
			#now post-process the classification
			#is the domain in domainRules?
			print u"({1}.{2} seconds) Top categories: {0}".format(u', '.join(['-'.join([unicode(y) for y in x]) for x in classification]), delta.seconds, delta.microseconds)
			print "---"
	else:
		print "Importing payload...",
		o = Ontology('payload.lwca')
		print "building index...."
		o.build_inverse_index()
		print "done."
		title = "MA Coin stores with 500.000 Coins - Medals - German Coins - Ancient Coins, World Coins Coins Mall"
		print "Classifying a webpage with title: \"{0}\"".format(title)
		print "--- with normal method:"
		start = datetime.now()
		classification = o.classify_webpage("", title)
		end = datetime.now()
		delta = end-start
		print u"({1}.{2} seconds) Top categories: {0}".format(u', '.join(['-'.join([unicode(y) for y in x]) for x in classification]), delta.seconds, delta.microseconds)
		print "--- now trying with ii:"
		start = datetime.now()
		classification = o.classify_webpage_with_ii(title)
		end = datetime.now()
		delta = end-start
		print u"({1}.{2} seconds) Top categories: {0}".format(u', '.join(['-'.join([unicode(y) for y in x]) for x in classification]), delta.seconds, delta.microseconds)
		print "---"
		while True:
			title = raw_input("Title: ")
			print "--- with normal method:"
			start = datetime.now()
			classification = o.classify_webpage("", title)
			end = datetime.now()
			delta = end-start
			print u"({1}.{2} seconds) Top categories: {0}".format(u', '.join(['-'.join([unicode(y) for y in x]) for x in classification]), delta.seconds, delta.microseconds)
			print "--- now trying with ii:"
			start = datetime.now()
			classification = o.classify_webpage_with_ii(title)
			end = datetime.now()
			delta = end-start
			print u"({1}.{2} seconds) Top categories: {0}".format(u', '.join(['-'.join([unicode(y) for y in x]) for x in classification]), delta.seconds, delta.microseconds)
			print "---"

