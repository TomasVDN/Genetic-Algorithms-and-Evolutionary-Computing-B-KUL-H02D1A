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

		yourConvergenceTestsHere = True
		while( yourConvergenceTestsHere ):
			# Your code here.

			parent1 = self.kTournamentSelection(5)
			parent2 = self.kTournamentSelection(5)
			self.generateSuccessorList(parent1, parent2)

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
		print(self.population)
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

	def generateSuccessorList(self, parent1: np.array, parent2: np.array):
		"""
		Create a n x 2 matrix, with the row index representing the city index and the column index representing the city that follows that city in parent1 and parent2.
		"""
		print(parent1)
		