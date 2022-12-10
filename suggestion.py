generateNearestNeighbourList()

population = []
while len(population) < populationSize:
    startCity = selectRandomCity()
    solution = [startCity]

    while len(visitedCities) < numberOfCities:
        # Find the nearest neighbor not already visited.
        for neighbor in nearestNeighbourList[currentCity]:
            if neighbor not in visitedCities:
                solution += [neighbor]
                currentCity = neighbor
                break
        else:
            solution = None

    if solution is not None:
        population += [solution]