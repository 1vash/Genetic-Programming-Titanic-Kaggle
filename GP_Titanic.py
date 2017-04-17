import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def Outputs(data):
	return np.round(1.-(1./(1.+np.exp(-data)))) #ans 1 | 0


def GP_deap(evolved_train):

	import operator
	import math
	import random


	from deap import algorithms
	from deap import base, creator
	from deap import tools
	from deap import gp


	inputs = evolved_train.iloc[:,2:10].values.tolist() 
	outputs = evolved_train['Survived'].values.tolist()


	def protectedDiv(left, right):
		try:
			return left / right
		except ZeroDivisionError:
			return 1


	#choosing Primitives
	pset = gp.PrimitiveSet("MAIN", 8) 
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(protectedDiv, 2)
	pset.addPrimitive(math.cos, 1)
	pset.addPrimitive(math.sin, 1)
	pset.addPrimitive(math.tanh,1)
	pset.addPrimitive(max, 2)
	pset.addPrimitive(min, 2)
	pset.addEphemeralConstant("rand101", lambda: random.uniform(-10,10))

	pset.renameArguments(ARG0='x1')
	pset.renameArguments(ARG1='x2')
	pset.renameArguments(ARG2='x3')
	pset.renameArguments(ARG3='x4')
	pset.renameArguments(ARG4='x5')
	pset.renameArguments(ARG5='x6')
	pset.renameArguments(ARG6='x7')
	pset.renameArguments(ARG7='x8')

	# two object types is needed: an individual containing the genotype
	# and a fitness -  The reproductive success of a genotype (a measure of quality of a solution)
	creator.create("FitnessMin", base.Fitness, weights=(1.0,))
	creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


	#register some parameters specific to the evolution process.
	toolbox = base.Toolbox()
	toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) #
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("compile", gp.compile, pset=pset)


	#evaluation function, which will receive an individual as input, and return the corresponding fitness.
	def evalSymbReg(individual):
		# Transform the tree expression in a callable function
		func = toolbox.compile(expr=individual)
		# Evaluate the accuracy of individuals // 1|0 == survived
		return math.fsum(np.round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs)) / len(evolved_train),


	toolbox.register("evaluate", evalSymbReg)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
	toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

	toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
	toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

	pop = toolbox.population(n=300)
	hof = tools.HallOfFame(1)

	#Statistics over the individuals fitness and size
	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)


	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=100, stats=stats,
								   halloffame=hof, verbose=True)

	#Parameters:
	#population – A list of individuals.
	#toolbox – A Toolbox that contains the evolution operators.
	#cxpb – The probability of mating two individuals.
	#mutpb – The probability of mutating an individual.
	#ngen – The number of generation.
	#stats – A Statistics object that is updated inplace, optional.
	#halloffame – A HallOfFame object that will contain the best individuals, optional.
	#verbose – Whether or not to log the statistics.

	# Transform the tree expression of hof[0] in a callable function and return it
	func2 = toolbox.compile(expr=hof[0]) 

	return func2


def MungeData(data):

	title_list = [
				'Dr', 'Mr', 'Master',
				'Miss', 'Major', 'Rev',
				'Mrs', 'Ms', 'Mlle','Col',
				'Capt', 'Mme', 'Countess',
				'Don', 'Jonkheer'
								]

	#replacing all people's name by their titles
	def replace_names_titles(x):
		for title in title_list:
			if title in x:
				return title

	data['Title'] = data.Name.apply(replace_names_titles)
	data['Title'] = data.Title.map({ 'Dr':1, 'Mr':2, 'Master':3, 'Miss':4, 'Major':5, 'Rev':6, 'Mrs':7, 'Ms':8, 'Mlle':9,
					 'Col':10, 'Capt':11, 'Mme':12, 'Countess':13, 'Don': 14, 'Jonkheer':15
					})

	data.drop(['Name'], 1, inplace=True)


	# Age
	data.Age.fillna(value=data.Age.mean(), inplace=True)
	# Relatives
	data['Relatives'] = data.SibSp + data.Parch
	# Fare per person
	data['Fare_per_person'] = data.Fare / np.mean(data.SibSp + data.Parch + 1)
	data.drop(['SibSp', 'Parch'], inplace=True, axis=1)
	data.drop(['Fare'], inplace=True, axis=1)
	# Ticket
	data.drop(['Ticket'], inplace=True, axis=1)
	# Sex
	data.Sex.fillna('0', inplace=True)
	data.loc[data.Sex != 'male', 'Sex'] = 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	# Cabin
	data.Cabin.fillna('0', inplace=True)
	data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
	data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
	data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
	data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
	data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
	data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
	data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
	data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
	# Embarked
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 3
	data.Embarked.fillna(0, inplace=True)
	data.fillna(-1, inplace=True)

	return data.astype(float)


if __name__ == "__main__":
	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	evolved_train = MungeData(train)
	evolved_test = MungeData(test)

	#GF
	GeneticFunction = GP_deap(evolved_train)

	#train
	train = evolved_train.iloc[:,2:10].values.tolist()
	# Evaluate the accuracy of h[0] on train set
	trainPredictions = Outputs(np.array([GeneticFunction(*x) for x in train]))

	pd_train = pd.DataFrame({'PassengerId': evolved_train.PassengerId.astype(int),
							'Predicted': trainPredictions.astype(int),
							'Survived': evolved_train.Survived.astype(int)})
	pd_train.to_csv('gptrain.csv', index=False)

	print(accuracy_score(evolved_train.Survived.astype(int),trainPredictions.astype(int)))

	#test
	test = evolved_test.iloc[:,1:9].values.tolist()
	# Evaluate the accuracy of h[0] on test set
	testPredictions = Outputs(np.array([GeneticFunction(*x) for x in test]))

	pd_test = pd.DataFrame({'PassengerId': evolved_test.PassengerId.astype(int),
							'Survived': testPredictions.astype(int)})
	pd_test.to_csv('gptest.csv', index=False)
