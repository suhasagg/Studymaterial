#!/usr/bin/env python

"""
Python implementation of DFR style classifier

Partly copied from : https://raw.githubusercontent.com/mzhilyaev/pfeed/master/stats/DFRClassifier.js
but with several upgrades/modifications

Usage:
> import mozclassify
> DFR = mozclassify.DFR()
> DFR.classify("http://www.coinweek.com/us-coins/the-marvelous-pogue-family-coin-collection-part-2-the-oliver-jung-1833-half-dime/")
[u'hobbies & interests', u'coins']
"""

from json import load
from urlparse import urlparse
from re import findall
from tldextract import extract
from collections import Counter, defaultdict

class DFR:
	"""Object that can classify URLs using a domain-rule based approach"""
	
	def __init__(self):
		"""Import DFR payload"""
		with open('payload_DFR_HTL.json') as f:
			self.dfr = load(f)
		
		self.kSplitter = "[\s-]+"

	def interestFinalizer(self, interests):
		#This makes a decision between the interests presented to it
		#Completely modified from pfeed version
		
		if len(interests) == 0: #if there's no result, just return uncategorized
			return ['uncategorized', 'unknown']
		elif len(interests) == 1:
			return interests[0]
		else:
			
			#Count up tallies for [top, sub] pairs
			finalInterests = defaultdict(int)
			for interest in interests:
				finalInterests[tuple(interest)] += 1
			finalInterests = sorted(finalInterests.items(), key=lambda x: x[1], reverse=True)
			
			if len(finalInterests) == 1: #if there's one result then return it
				return finalInterests[0][0]
			elif finalInterests[0][1] > finalInterests[1][1]: #is the top item better than the second item?
				return finalInterests[0][0] #if so, just return the top item
			else:
				#is there a top level consensus, e.g.:
				# 1) ((sports, baseball), 2)
				# 2) ((sports, soccer), 2)
				# 3) ((religion, islam), 2)
				# 4) ((folklore, astrology), 1)
				# in this case, we could return (sports, general)
				
				#make a dictionary like score: [classifications]
				scores = defaultdict(list)
				for interest in finalInterests:
					scores[interest[1]].append(interest[0])
				
				#now have something like:
				# 2: [(sports, baseball), (sports, soccer), (religion, islam)]
				# 1: [(folklore, astrology)]
				scores = sorted(scores.items(), reverse=True)[0][1] #grab the top item
				
				#Counter of the top level items
				top_levels = defaultdict(int)
				for decision in scores:
					top_levels[decision[0]] += 1
				
				top_levels = sorted(top_levels.items(), key=lambda x: x[1], reverse=True)
				
				# this would produce:
				# sports: 2
				# religion: 1
				
				if len(top_levels) == 1:
					return [top_levels[0][0], "general"]
				else:
					if top_levels[0][1] > top_levels[1][1]:
						return [top_levels[0][0], "general"]
					else:
						return ["uncategorized", "no consensus"]
	
	def convertVisittoDFR(self, host, baseDomain, path, title, url):
		"""Finds words and bigrams contained within the URL and title. Outputs them in a set with appropriate suffixes."""
		
		words = set()
		
		def addToWords(chunks, options={}):
			"""this function populates the words object with terms
				It adds the apropriate suffix (it case of host chunks)
				or prefix (in case of paths) to the chunks supplied"""
			
			if "prefix" not in options: options['prefix'] = ""
			if "suffix" not in options: options['suffix'] = ""
			if "clearText" not in options: options['clearText'] = False
			
			prev = ""
			for chunk in chunks: 
				words.update([options['prefix'] + chunk + options['suffix']])
				if options['clearText']:
					words.update([chunk])
				if prev: #add bigram
					words.update([options['prefix'] + prev + chunk + options['suffix']])
					if options['clearText']:
						words.update([prev + chunk])
				prev = chunk
		
		# tokenize and add title chunks
		addToWords(self.tokenize(title), {"suffix": "_t", "clearText": True})
		
		# tokenize and add url chunks
		addToWords(self.tokenize(url), {"suffix": "_u", "clearText": True})
		
		# parse and add hosts chunks
		hostChunks = host.split(".")
		addToWords(hostChunks, {"suffix": "."})
		
		# add subdomains under __SCOPED keyword
		scopedHosts = [baseDomain]
		hostString = baseDomain
		
		for chunk in hostChunks:
			hostString = ".".join([chunk, hostString])
			scopedHosts.append(hostString)
		
		# parse and add path chunks
		pathChunks = path.split("/")
		
		for chunk in pathChunks:
			addToWords(self.tokenize(chunk), {"prefix": "/"})
		
		return [words, scopedHosts]
	
	def tokenize(self, s):
		"""Tokenizes a string"""
		return findall("[a-z]{3,}", s.lower())

	def classify(self, url, title="", rules_only=False):
		"""Classifies a url (and possibly the title)"""
		
		#parse the url
		parsed_url = urlparse(url);
		tld = extract(url)
		
		#get the specific components
		baseDomain = ".".join([tld.domain, tld.suffix])
		host = tld.subdomain
		path = parsed_url.path
		
		#setup
		self.interests = []
		
		# check if rules are applicable at all
		if baseDomain not in self.dfr and "__ANY" not in self.dfr:
			return self.interests
		
		# populate words object with visit data
		ret = self.convertVisittoDFR(host, baseDomain, path, title, url)
		words = ret[0]
		scopedHosts = ret[1]
		
		def matchedAllTokens(tokens):
			"""this function tests for existence of rule terms in the words object
			if all rule tokens are found in the words object return true"""
			if len(set(tokens).intersection(words)) == len(tokens):
				return True
			else:
				return False
		
		def matchRuleInterests(rule):
			"""match a rule and collect matched interests"""
			for key in rule.iterkeys():
				if (key == "__HOME" and (path == null or path == "" or path == "/" or path.startswith("/?"))):
					self.interests += rule[key]
				else:
					if ("__" not in key and matchedAllTokens(findall(self.kSplitter, key))):
						self.interests += rule[key]
		
		def matchANYRuleInterests(rule):
			""" __ANY rule does not support multiple keys in the rule
				__ANY rule matches any single term rule - but not the term combination
				as in "/foo bar_u baz_t"
			"""
			for word in words:
				if word in rule:
					self.interests += rule[word]
		
		def isWhiteListed(hosts, whiteList):
			"""checks if any of the provided scoped hosts are white listed"""
			for i in hosts:
				if i in whiteList:
					return True
			return False
		
		# process __ANY rule first
		if not rules_only:
			if (self.dfr["__ANY"]):
				matchANYRuleInterests(self.dfr["__ANY"])
		
		if not rules_only:
			if self.dfr["__SCOPES"]:
				#dfr has scoped rules - check for scope domains and sub-domains
				#check if scopedHosts are white-listed in any of the __SCOPED rule
				#and if so apply the rule
				
				for scopedRule in self.dfr["__SCOPES"]:
					# the scopedRule is of the form {"__HOSTS": {"foo.com", "bar.org"}, "__ANY": {... the rule...}}
					if isWhiteListed(scopedHosts, scopedRule["__HOSTS"]):
						matchANYRuleInterests(scopedRule["__ANY"])
						# we do not expect same page belong to two different genre
						break
		
		if baseDomain in self.dfr:
			domainRule = self.dfr[baseDomain]
			
			keyLength = len(domainRule) if domainRule else 0
			
			if not keyLength:
				return this.interestFinalizer(self.interests)
			
			if "__ANY" in domainRule:
				self.interests += domainRule["__ANY"]
				keyLength -= 1
			
			if not keyLength:
				return self.interestFinalizer(self.interests)
			
			matchRuleInterests(domainRule)
		
		return self.interestFinalizer(self.interests)

