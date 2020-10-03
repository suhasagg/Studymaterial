#make new payload

#Test out the algorithms for their precision and recall
#Usage:
# >>> from test import test_algorithms
# >>> test_algorithms()
# 
# (results printed to terminal)
#

from pymongo import Connection
from json import load
from collections import defaultdict

from classifier_LICA import LICA
from classifier_DFR import DFR

def check_mappings():
	"""	Moreover uses a different set of topic names than in mozcat.
		These have been mapped in moreover_to_mozcat.json
		However, mozcat_heirarchy changes fairly often so we have to check that this is still up to date
		can be checked easily with:
		python -c "from test import check_mappings;print check_mappings()"
	"""
	
	#load the files into memory
	with open('/Users/mruttley/Documents/2015-01-13 Heirarchy/mozcat/mozcat_heirarchy.json') as f: 
		# tree is currently in the format:
		# top_level: [sub_level, sub_level, ...]
		# this is hard to look up from, so we need a set of all items which is O(1)
		mozcat = set()
		for top_level, sub_levels in load(f).iteritems():
			mozcat.update([top_level])
			mozcat.update(sub_levels)
		
	with open('moreover_to_mozcat.json') as f:
		mapping = load(f)
	
	not_found = []
	for k, v in mapping.iteritems():
		if v not in mozcat:
			not_found.append(v)
	
	if not_found:
		print "Not Found: {0}".format(sorted(list(set(not_found))))
		return False
	else:
		return True

def test_functionality():
	"""	Tests the output of each algorithm for specific URLs, to make sure that they are returning correctly
		Easy to use by doing: python -c "import test;test.test_functionality()"
	"""
		
	#set things up
	algorithms = [
		{
			"name": 'LICA',
			"function_object": LICA(),
			"correct": 0,
			"incorrect": 0
		},
		{
			"name": 'DFR',
			"function_object": DFR(),
			"correct": 0,
			"incorrect": 0
		},
	]
	
	tests = [
		{
			"url": "https://advocacy.mozilla.org/open-web-fellows/",
			"title": "Ford-Mozilla Open Web Fellows | Mozilla Advocacy",
			"expectedResult": ('technology & computing', 'general')
		},
		{
			"url": "http://www.budget-cruises.com/cruise/maldives",
			"title": "Budget Cruises to the Maldives",
			"expectedResult": ('travel', 'cruise vacations')
		}
	]
	
	correct = 0
	incorrect = 0
	for test in tests:
		print test['url']
		print test['title']
		for a in range(len(algorithms)):
			result = algorithms[a]['function_object'].classify(test['url'])
			result = tuple(result)
			if result == test['expectedResult']:
				print "{0}: Correctly classified this as: {1}".format(algorithms[a]['name'], result)
				algorithms[a]['correct'] += 1
			else:
				print "{0}: Incorrectly classified as: {1} when it should be {2}".format(algorithms[a]['name'], result, test['expectedResult'])
				algorithms[a]['incorrect'] += 1

def output_stats(results):
	"""Outputs some stats from a results object in test_algorithms"""
	
	total = sum(results[results.keys()[0]].values())
	print "Total documents tested: {0}".format(total)
	
	for algorithm, tallies in results.iteritems():
		print "    Algorithm: {0}".format(algorithm.upper())
		print "      Correct: {0}".format(tallies['correct'])
		print "    Incorrect: {0}".format(tallies['incorrect'])
		print "Uncategorized: {0}".format(tallies['uncategorized'])
		print "    Precision: {0}".format(round((tallies['correct']/float(tallies['correct']+tallies['incorrect']))*100, 3) if tallies['incorrect'] > 0 else 0)
		print "       Recall: {0}".format(round((tallies['correct']+tallies['incorrect'])/float(total)*100, 3) if total > 0 else 0)
	
	print "-"*50

def build_roc_curve_data(file_name):
	"""	Creates ROC curve data from the output of test_algorithms
		Note: Only works for binary classifiers
		The best way to input data from this is to do:
		> python -c "import test;test.test_algorithms()" > some_file.txt
		> python -c "import test;test.build_roc_curve_data()"
		"""

	with open(file_name) as f:
		
		#will be stored like:
		#{
		#	'Some classifier': [
		#		[tp, fp], [tp, fp]
		#	]
		#}
		algorithms = defaultdict(list)
		
		#have to load the entire file into memory
		for status in f.read().split("--------------------------------------------------"):
			status = status.strip() 
			if status:
				#clean up each status
				status = [x.strip() for x in status.split("\n") if x.strip() != ""][1:]
				
				for x in range(len(status)/6): #currently testing 6 algos
					algo = status[6*x:(x*6)+6]
					
					correct = 0
					incorrect = 0
					name = ""
					
					for metric in algo:
						if "Algorithm" in metric:
							name = metric.split(": ")[1]
						if "Correct" in metric:
							correct = int(metric.split(": ")[1])
						if "Incorrect" in metric:
							incorrect = int(metric.split(": ")[1])
					
					#print line
					#print correct, incorrect, name
					
					if correct + incorrect == 0:
						true_positive_rate = 0
						false_positive_rate = 0
					else:
						true_positive_rate = float(correct) / (correct + incorrect)
						false_positive_rate = 1 - true_positive_rate
					
					algorithms[name].append([true_positive_rate, false_positive_rate])
		
		with open("roc.csv", 'w') as f:
			for k,v in algorithms.iteritems():
				f.write("{0}\t{1}\n".format(k+"_tp", k+"_fp"))
				for x in v:
					f.write("{0}\t{1}\n".format(x[0], x[1]))

def create_connection():
	"""Creates and returns a connection to MongoDB"""
	return Connection("ec2-54-87-201-148.compute-1.amazonaws.com")['moreover']['docs']

def create_lookup_trees():
	"""Creates a tree, reverse_tree, mozcat_to_moreover, moreover_to_mozcat"""
	
	#load the moreover to mozcat mappings (moreover uses a different dataset)
	with open("moreover_to_mozcat.json") as mm:
		with open("/Users/mruttley/Documents/2015-01-13 Heirarchy/mozcat/mozcat_heirarchy.json") as mh:
			#bit tricky as the mapping file just gives the top/sub level, not a [top, sub] pair as needed
			#so we have to convert it
			tree = load(mh)
			
			#build a reverse tree of sub_level = [top_level, sub_level]
			#for easy lookups
			reverse_tree = {}
			for k,v in tree.iteritems():
				reverse_tree[k] = [k, "general"]
				for x in v:
					reverse_tree[x] = [k, x]
				
			#now make sure the mappings point towards those pairs rather than strings
			#the most useful format is mozcat to moreover
			mozcat_to_moreover = defaultdict(set)
			moreover_to_mozcat = {}
			for k, v in load(mm).iteritems():
				mozcat_to_moreover[tuple(reverse_tree[v])].update([k])
				moreover_to_mozcat[k] = reverse_tree[v]
	
	return [tree, reverse_tree, mozcat_to_moreover, moreover_to_mozcat]


def test_algorithms(top_only=False):
	"""Tests the algorithms on Moreover data"""
	
	#set up the connection and initialize the classifiers
	db = create_connection()
	lica = LICA()
	dfr = DFR()
	
	tree, reverse_tree, mozcat_to_moreover, moreover_to_mozcat = create_lookup_trees()
	
	#something to store the results in
	results = defaultdict(lambda: {
			"correct": 0,
			"incorrect": 0,
			"uncategorized": 0
		})
	
	#iterate through the documents
	for n, document in enumerate(db.find({'topics': {'$exists':True}}, {'url':1, 'topics':1, 'title':1, 'content':1})):
		
		#classify (if they are in an object like this, they are easier to process)
		decisions = {
			'dfr': dfr.classify(document['url']),
			'lica': lica.classify(document['url']),
			'dfr_title': dfr.classify(document['url'], title=document['title']),
			'lica_title': lica.classify(document['url'], title=document['title']),
			'dfr_content': dfr.classify(document['url'], title=document['content']),
			'lica_content': lica.classify(document['url'], title=document['content']),
			'dfr_title_rules_only': dfr.classify(document['url'], title=document['title'], rules_only=True),
			'dfr_rules_only': dfr.classify(document['url'], rules_only=True),
		}
		
		for algorithm, decision in decisions.iteritems():
			decision = tuple(decision) #have to be tuples for the mappings dictionary
			if decision[0] == 'uncategorized':
				results[algorithm]['uncategorized'] += 1
			else:
				if top_only:
					
					decision = decision[0] #get top level
					topics = set([moreover_to_mozcat[x][0] for x in document['topics'] if x in moreover_to_mozcat])
					
					if decision in topics:
						results[algorithm]['correct'] += 1
					else:
						results[algorithm]['incorrect'] += 1
				else:
					decision = mozcat_to_moreover[decision]
					topics = set(document['topics'])
					
					if decision.intersection(topics):
						results[algorithm]['correct'] += 1
					else:
						results[algorithm]['incorrect'] += 1
		
		#output some stats occasionally
		if n % 10000 == 0:
			output_stats(results)
	
	print "Classification testing finished, final results:"
	output_stats(results)