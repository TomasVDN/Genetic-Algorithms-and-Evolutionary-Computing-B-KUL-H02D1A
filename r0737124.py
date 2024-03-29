import Reporter
import numpy as np

# Modify the class name to match your student number.
class r0737124:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

		self.numberOfCities = 0
		self.distanceMatrix: np.array = None

		self.nearestNeighbourList = None

		self.populationSize = 25

		self.mutationChance = 0.05

		# Selection of the crossover method. Only one of these should be True.
		self.useSimpleCrossover = True
		self.useSequentialConstructiveCrossover = False


	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		self.distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		print(f'Starting initialization...')

		self.numberOfCities = len(self.distanceMatrix)
		print(f'Number of cities: {self.numberOfCities}')

		self.calculateNearestNeighboursList()
		print(f'Nearest neighbours list calculated.')

		population = self.initializePopulation()
		fitness = self.calculatePopulationFitness(population)
		print(f'Population initialized.')

		print(f'Initialization done.')
		print(f'Starting optimization...')


		meanObjective = 0.0
		bestObjective = 0.0
		bestSolution = np.array([])

		generation = 0

		yourConvergenceTestsHere = True
		while( yourConvergenceTestsHere ):
			# Your code here.
			# Mutate population.
			population = self.reverseSequenceMutate(population)

			# Create children.
			children = self.createChildren(population, fitness)

			# Perform optimization and selection: append children to population, run local optimizer and select the populationSize best solutions.
			population, fitness = self.optimizeAndSelect(population, children)

			# Normalize: make all candidates start with 0.
			population = self.normalize(population)

			# Update best solution, mean objective and best objective.
			bestSolutionIndex = np.argmin(fitness)
			bestSolution = population[bestSolutionIndex]
			bestObjective = fitness[bestSolutionIndex]
			meanObjective = np.mean(fitness)

			# Report the best solution, mean objective and best objective.
			# print(f'Best solution: {bestSolution}')
			print(f'Mean objective: {meanObjective}')
			print(f'Best objective: {bestObjective}')

			generation += 1
			print(f'Generation: {generation}')


			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			
		# Your code here.
		print('Optimization done.')
		return 0

	def optimizeAndSelect(self, population, children):
		"""
		Append children to population, and select the populationSize best solutions.
		"""
		population = np.append(population, children, axis=0)
		fitness = self.calculatePopulationFitness(population)

		population, fitness = self.runLocalOptimizer(population, fitness)

		population = population[np.argsort(fitness)][:self.populationSize]
		fitness = self.calculatePopulationFitness(population)
		return population,fitness

	def createChildren(self, population, fitness):
		"""
		Create children using the given population and fitness. The children are created using either simple crossover or sequential constructive crossover.
		"""
		children = np.empty((0, self.numberOfCities), int)
				
		if (self.useSimpleCrossover):
			while len(children) < self.populationSize:
				parent1 = self.kTournamentSelection(5, fitness)
				parent2 = self.kTournamentSelection(5, fitness)
				child1, child2 = self.crossover(parent1, parent2, population)
				children = np.append(children, [child1], axis=0)
				children = np.append(children, [child2], axis=0)

		elif (self.useSequentialConstructiveCrossover):
			while len(children) < self.populationSize:
				parent1 = self.kTournamentSelection(5, fitness)
				parent2 = self.kTournamentSelection(5, fitness)
				child = self.sequentialConstructiveCrossover(parent1, parent2, population)
				children = np.append(children, [child], axis=0)

		return children

	def calculateNearestNeighboursList(self):
		"""
		For each city in distanceMatrix, sort the list of indices of the other cities in function of 
		the distance relative to the current city. Remove all indices for which the distance is 0 of inf.
		"""
		nearestNeighboursList = []
		for i in range(self.numberOfCities):
			nearestNeighboursList.append(np.argsort(self.distanceMatrix[i]))
			nearestNeighboursList[i] = np.delete(nearestNeighboursList[i], np.where(self.distanceMatrix[i][nearestNeighboursList[i]] == 0))
			nearestNeighboursList[i] = np.delete(nearestNeighboursList[i], np.where(np.isinf(self.distanceMatrix[i][nearestNeighboursList[i]])))

		self.nearestNeighbourList = nearestNeighboursList
			
	def greedySearch(self, startCityIndex: int):
		"""
		Starting from the given startCityIndex, add the nearest neighbor using self.nearestNeighbourList not already in the visited cities. Continue until all 
		cities are visited. If the city for which a neighbor must be found does not have any unvisited neighbors, return None.
		"""
		visitedCities = np.array([startCityIndex])
		currentCity = startCityIndex
		while len(visitedCities) < self.numberOfCities:
			# Find the nearest neighbor not already visited.
			for neighbor in self.nearestNeighbourList[currentCity]:
				if neighbor not in visitedCities:
					visitedCities = np.append(visitedCities, neighbor)
					currentCity = neighbor
					break
			else:
				return None

		return visitedCities

	def initializePopulation(self):
		""""
		Initialize the first generation. Until the population reaches the given threshold, generate a random starting city and perform a greedy search to find
		a solution. Add the solution to the population.
		"""
		population = np.empty((0, self.numberOfCities), int)
		while len(population) < self.populationSize:
			startCityIndex = np.random.randint(self.numberOfCities)
			solution = self.greedySearch(startCityIndex)
			if solution is not None:
				population = np.append(population, [solution], axis=0)

		return population

	def calculateFitness(self, solution: np.array):
		"""
		Calculate the fitness of the given solution.
		"""
		fitness = 0
		for i in range(len(solution) - 1):
			fitness += self.distanceMatrix[solution[i]][solution[i + 1]]
		fitness += self.distanceMatrix[solution[-1]][solution[0]]

		return fitness


	def calculatePopulationFitness(self, population: np.array):
		"""
		Calculate the fitness of each solution in the population.
		"""
		return np.apply_along_axis(self.calculateFitness, 1, population)


	def kTournamentSelection(self, k: int, fitness: np.array):
		"""
		Select k random solutions from the population and return the index of the best one.
		"""
		solutions = np.random.randint(self.populationSize, size=k)
		bestSolution = solutions[0]
		for solution in solutions:
			if fitness[solution] < fitness[bestSolution]:
				bestSolution = solution

		return bestSolution

	def crossover(self, parent1Index: int, parent2Index: int, population: np.array):
		"""
		Select a random index. Copy the first part of a parent to a child, and the first part of the second parent to the second child. Then, for each child, 
		add the cities of the other parent in the order they appear in the other parent, skipping the cities already in the child.
		"""
		child1 = np.empty((0), int)
		child2 = np.empty((0), int)

		index = np.random.randint(self.numberOfCities)

		child1 = np.append(child1, population[parent1Index][:index])
		child2 = np.append(child2, population[parent2Index][:index])

		for city in population[parent2Index]:
			if city not in child1:
				child1 = np.append(child1, city)

		for city in population[parent1Index]:
			if city not in child2:
				child2 = np.append(child2, city)

		return child1, child2

	def sequentialConstructiveCrossover(self, parent1Index: int, parent2Index: int, population: np.array):
		"""
		Creates a child with one city: 0. Then, search both parents for next unused neighbor of the last city added to the child (with the findNextUnusedNode function)
		and add it to the child. This is repeated until all cities are added to the child.
		"""
		child = np.array([0])
		visitedCities = np.array([0])

		parent1 = population[parent1Index]
		parent2 = population[parent2Index]

		while len(child) < self.numberOfCities:
			city1 = self.findNextUnusedNode(child[-1], parent1, visitedCities)
			city2 = self.findNextUnusedNode(child[-1], parent2, visitedCities)

			# Add the nearest city to the child.
			if self.distanceMatrix[child[-1]][city1] < self.distanceMatrix[child[-1]][city2]:
				child = np.append(child, city1)
				visitedCities = np.append(visitedCities, city1)
			else:
				child = np.append(child, city2)
				visitedCities = np.append(visitedCities, city2)

		return child
			
	def findNextUnusedNode(self, previousNode: int, parent: np.array, alreadyUsedNodes: np.array):
		"""
		Find the index of the given previousNode in the parent. Then, search in the nodes after that index for the first node not in alreadyUsedNodes
		(due to the cyclic nature, restart at the beginning of the parent if the end is reached). If no node is found, return None.
		"""
		index = np.where(parent == previousNode)[0][0]
		for node in parent[index + 1:]:
			if node not in alreadyUsedNodes:
				return node
			
		for node in parent[:index + 1]:
			if node not in alreadyUsedNodes:
				return node

		return None


	def reverseSequenceMutate(self, children: np.array):
		"""
		For each child in the list of children, decide if it must be mutated. If it must be mutated, select two random indices and reverse the sequence of nodes between them.
		"""
		for i in range(len(children)):
			if np.random.rand() < self.mutationChance:
				startIndex = np.random.randint(self.numberOfCities)
				endIndex = np.random.randint(self.numberOfCities)
				startIndex, endIndex = min(startIndex, endIndex), max(startIndex, endIndex)
				children[i][startIndex:endIndex] = children[i][startIndex:endIndex][::-1]

		return children

	def normalize(self, population: np.array):
		"""
		Shift all cycles in the population such that the first element is 0.
		"""
		for i in range(len(population)):
			index = np.where(population[i] == 0)[0][0]
			population[i] = np.roll(population[i], -index)

		return population


	def removeDuplicates(self, population: np.array):
		"""
		Remove duplicate solutions from the population.
		"""
		return np.unique(population, axis=0)


	def runLocalOptimizer(self, population: np.array, fitness: np.array):
		"""
		For each solution in the population, run the local optimizer.
		"""
		for i in range(len(population)):
			population[i], fitness[i] = self.naiveOptimization(population[i], fitness[i])

		return population, fitness


	def naiveOptimization(self, candidate: np.array, currentFitness: float):
		"""
		For each node in the candidate, swap it with the next node and calculate the fitness of the candidate.
		If the fitness is better than the current fitness, the candidate and the fitness are updated.
		"""
		for i in range(len(candidate) - 1):
			candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
			newFitness = self.calculateFitness(candidate)
			if newFitness < currentFitness:
				currentFitness = newFitness
			else:
				candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]

		return candidate, currentFitness


		