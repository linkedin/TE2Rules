class RuleBuilder:
	def __init__(self, random_forest, X, y):
		self.random_forest = random_forest
		self.labels = y
		self.positives = []
		for i in range(len(self.labels)):
			if(self.labels[i] == 1):
				self.positives.append(i)

		print()
		print("Positives: " + str(len(self.positives)))

		print()
		print("Rules from trees")
		self.candidate_rules = random_forest.get_rules(data=X)
		self.solution_rules = []
		print(str(len(self.candidate_rules)) + " candidate rules")
		
		self.deduplicate()
		
		print("Deduping...")
		print(str(len(self.candidate_rules)) + " candidate rules")
		
		self.generate_solutions()

		print()
		print("Set Cover")
		self.pick_top_rules()

	def pick_top_rules(self):
		self.positives_to_explain = self.positives		
		
		positive_support = []
		for r in self.solution_rules:
			positive_support.append(list(set(self.positives).intersection(set(r.decision_support))))

		top_rule_indices = []
		while(len(self.positives_to_explain) > 0):
			#choose biggest one
			index = 0
			for i in range(len(positive_support)):
				if(len(positive_support[i]) > 0):
					if(len(positive_support[i]) > len(positive_support[index])):
						index = i
					else:
						if(len(positive_support[i]) == len(positive_support[index])):
							if(len(self.solution_rules[i].decision_rule) < len(self.solution_rules[index].decision_rule)):
								index = i

			top_rule_indices.append(index)
			self.positives_to_explain = list(set(self.positives_to_explain).difference(set(positive_support[index])))

			#remove support
			for i in range(len(positive_support)):
				if(index != i):
					positive_support[i] = list(set(positive_support[i]).difference(set(positive_support[index])))
			positive_support[index] = []
		
		solutions = []
		for i in top_rule_indices:
			solutions.append(self.solution_rules[i])
		self.solution_rules = solutions
		print(len(self.solution_rules))


	def generate_solutions(self):
		print()
		print("Running Apriori")

		self.positives_to_explain = self.positives
		stage = 0
		while((len(self.positives_to_explain) > 0) and (stage <= self.random_forest.get_num_trees())):
			print()
			print("Rules from " + str(stage + 1) + " trees")
		
			new_candidates = []
			if(stage == 0):
				for r in self.candidate_rules:
					if(self.filter_candidates(r) == True):
						new_candidates.append(r)
			else:
				for i in range(len(self.candidate_rules)):
					for j in range(i, len(self.candidate_rules)):
						r = self.candidate_rules[i].join(self.candidate_rules[j])
						if((r is not None) and (self.filter_candidates(r) == True)):
								new_candidates.append(r)
			self.candidate_rules = new_candidates

			self.prune()

			print("Pruning Candidates")
			print(str(len(self.candidate_rules)) + " candidates")
			print(str(len(self.solution_rules)) + " solutions")

			self.deduplicate()
			
			print("Deduping...")
			print(str(len(self.candidate_rules)) + " candidates")
			print(str(len(self.solution_rules)) + " solutions")
			
			fidelity, fidelity_positives, fidelity_negatives = self.get_fidelity()
			
			print("Fidelity")
			print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
			
			print("Unexplained Positives")
			print(len(self.positives_to_explain))

			stage = stage + 1


	def filter_candidates(self, rule):
		decision_value = []
		for data_index in rule.decision_support:
			decision_value.append(self.labels[data_index])

		decision_rule_precision = self.get_precision(rule, scores = decision_value)
		solution_is_possible = self.check_coverage(rule, scores = decision_value)

		if(decision_rule_precision >= 0.95):
			self.solution_rules.append(rule)
			self.positives_to_explain = list(set(self.positives_to_explain).difference(set(rule.decision_support)))
		else:
			if(solution_is_possible):
				return True
		return False


	def get_fidelity(self, use_top = None):
		if(use_top is None):
			use_top = len(self.solution_rules)

		support = []
		for i in range(use_top):
			r = self.solution_rules[i]
			support = support + r.decision_support
		support = list(set(support))

		y_pred_rules = [0]*len(self.labels)
		for s in support:
			y_pred_rules[s] = 1 
		
		positives = 0
		fidelity_positives = 0
		negatives = 0
		fidelity_negatives = 0
		for i in range(len(self.labels)):
			if(self.labels[i] == 1):
				positives = positives + 1
				if(y_pred_rules[i] == self.labels[i]):
					fidelity_positives = fidelity_positives + 1
			if(self.labels[i] == 0):
				negatives = negatives + 1
				if(y_pred_rules[i] == self.labels[i]):
					fidelity_negatives = fidelity_negatives + 1

		print(fidelity_positives, positives)
		print(fidelity_negatives, negatives)
		return (fidelity_positives + fidelity_negatives) / (positives + negatives), fidelity_positives / positives, fidelity_negatives / negatives


	def prune(self):
		pruned_candidates = []
		for i in range(len(self.candidate_rules)):
			rule = self.candidate_rules[i]

			decision_value = []
			for data_index in rule.decision_support:
				decision_value.append(self.labels[data_index])

			solution_is_possible = self.check_coverage(rule, scores = decision_value)
			
			if(solution_is_possible):
				pruned_candidates.append(rule)
		self.candidate_rules = pruned_candidates
		return

	def get_precision(self, rule, scores = None):
		min_score, max_score = self.random_forest.get_rule_score(rule.decision_rule)

		decision_rule_precision = 0.00
		if(min_score*self.random_forest.weight + self.random_forest.bias >= 0):
			decision_rule_precision = 1.00
		
		if(scores is not None):
			decision_rule_precision = sum(scores)/len(scores)

		return decision_rule_precision
		
	def check_coverage(self, rule, scores = None):
		min_score, max_score = self.random_forest.get_rule_score(rule.decision_rule)

		solution_is_possible = True 
		if(max_score*self.random_forest.weight + self.random_forest.bias < 0):
			solution_is_possible = False
				
		if(scores is not None):
			if(max(scores) == 0):
				solution_is_possible = False
		
			decision_support_positive = list(set(rule.decision_support).intersection(set(self.positives_to_explain)))
		
			if(len(decision_support_positive) == 0):
				solution_is_possible = False

		return solution_is_possible


	def deduplicate(self):		
		def dedup(rules):		
			dedup_rules = []
			dedup_decision_rules = []
			for i in range(len(rules)):
				if(rules[i].decision_rule not in dedup_decision_rules):
					dedup_rules.append(rules[i])
					dedup_decision_rules.append(rules[i].decision_rule)
				else:
					index = dedup_decision_rules.index(rules[i].decision_rule)
					dedup_rules[index].identity = list(set(dedup_rules[index].identity).union(set(rules[i].identity)))

			for i in range(len(dedup_rules)):
				dedup_rules[i].create_identity_map()
			return dedup_rules

		self.candidate_rules = dedup(self.candidate_rules)
		self.solution_rules = dedup(self.solution_rules)
		return
