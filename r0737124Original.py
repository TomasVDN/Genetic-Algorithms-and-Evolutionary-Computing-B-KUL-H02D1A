import Reporter
import numpy as np

# Modify the class name to match your student number.
class r0737124:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

		self.numberOfCities = 0
		self.distanceMatrix: np.array = None

		self.nearestNeighbourList = None

		self.populationSize = 200

		self.population: np.array = None
		self.populationFitness: np.array = None

		self.mutationChance = 0.05
		
		self.children: np.array = None

		# Selection of the crossover method. Only one of these should be True.
		self.useSimpleCrossover = True
		self.useSequentialConstructiveCrossover = False

		self.test = True

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

		self.initializePopulation()
		self.calculatePopulationFitness()
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
			self.children = np.empty((0, self.numberOfCities), int)

			# Create children.
			if (self.useSimpleCrossover):
				while len(self.children) < self.populationSize:
					parent1 = self.kTournamentSelection(5)
					parent2 = self.kTournamentSelection(5)
					child1, child2 = self.crossover(parent1, parent2)
					self.children = np.append(self.children, [child1], axis=0)
					self.children = np.append(self.children, [child2], axis=0)
			elif (self.useSequentialConstructiveCrossover):
				while len(self.children) < self.populationSize:
					parent1 = self.kTournamentSelection(5)
					parent2 = self.kTournamentSelection(5)
					child = self.sequentialConstructiveCrossover(parent1, parent2)
					self.children = np.append(self.children, [child], axis=0)

			
			# Mutate children.
			self.reverseSequenceMutate()

			# Perform selection: append children to population, and select the populationSize best solutions.
			self.population = np.append(self.population, self.children, axis=0)
			self.calculatePopulationFitness()
			self.population = self.population[np.argsort(self.populationFitness)][:self.populationSize]
			self.calculatePopulationFitness()

			# Normalize: make all candidates start with 0.
			self.normalize()

			# Update best solution, mean objective and best objective.
			bestSolutionIndex = np.argmin(self.populationFitness)
			bestSolution = self.population[bestSolutionIndex]
			bestObjective = self.populationFitness[bestSolutionIndex]
			meanObjective = np.mean(self.populationFitness)

			# Report the best solution, mean objective and best objective.
			print(f'Best solution: {bestSolution}')
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

		self.population = population

	def calculateFitness(self, solution: np.array):
		"""
		Calculate the fitness of the given solution.
		"""
		fitness = 0
		for i in range(len(solution) - 1):
			fitness += self.distanceMatrix[solution[i]][solution[i + 1]]
		fitness += self.distanceMatrix[solution[-1]][solution[0]]

		return fitness

	def calculatePopulationFitness(self):
		"""
		Calculate the fitness of each solution in the population.
		"""
		fitness = np.array([])
		for solution in self.population:
			fitness = np.append(fitness, self.calculateFitness(solution))

		self.populationFitness = fitness

	def kTournamentSelection(self, k: int):
		"""
		Select k random solutions from the population and return the index of the best one.
		"""
		solutions = np.random.randint(self.populationSize, size=k)
		bestSolution = solutions[0]
		for solution in solutions:
			if self.populationFitness[solution] < self.populationFitness[bestSolution]:
				bestSolution = solution

		return bestSolution

	def crossover(self, parent1Index: int, parent2Index: int):
		"""
		Select a random index. Copy the first part of a parent to a child, and the first part of the second parent to the second child. Then, for each child, 
		add the cities of the other parent in the order they appear in the other parent, skipping the cities already in the child.
		"""
		child1 = np.empty((0), int)
		child2 = np.empty((0), int)

		index = np.random.randint(self.numberOfCities)

		child1 = np.append(child1, self.population[parent1Index][:index])
		child2 = np.append(child2, self.population[parent2Index][:index])

		for city in self.population[parent2Index]:
			if city not in child1:
				child1 = np.append(child1, city)

		for city in self.population[parent1Index]:
			if city not in child2:
				child2 = np.append(child2, city)

		return child1, child2

	def sequentialConstructiveCrossover(self, parent1Index: int, parent2Index: int):
		"""
		Creates a child with one city: 0. Then, search both parents for next unused neighbor of the last city added to the child (with the findNextUnusedNode function)
		and add it to the child. This is repeated until all cities are added to the child.
		"""
		child = np.array([0])
		visitedCities = np.array([0])

		parent1 = self.population[parent1Index]
		parent2 = self.population[parent2Index]

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


	def reverseSequenceMutate(self):
		"""
		For each child in the list of children, decide if it must be mutated. If it must be mutated, select two random indices and reverse the sequence of nodes between them.
		"""
		for i in range(len(self.children)):
			if np.random.rand() < self.mutationChance:
				startIndex = np.random.randint(self.numberOfCities)
				endIndex = np.random.randint(self.numberOfCities)
				startIndex, endIndex = min(startIndex, endIndex), max(startIndex, endIndex)
				self.children[i][startIndex:endIndex] = self.children[i][startIndex:endIndex][::-1]

	def normalize(self):
		"""
		Shift all cycles in the population such that the first element is 0.
		"""
		for i in range(len(self.population)):
			index = np.where(self.population[i] == 0)[0][0]
			self.population[i] = np.roll(self.population[i], -index)
	


		