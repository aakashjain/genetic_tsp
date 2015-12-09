import math
import random
import imp


class City:
    # for random initializion, leave coordinates blank
    def __init__(self, x=None, y=None):
        if x is not None:
            self.x = x
        else:
            self.x = int(random.random() * 500)
        if y is not None:
            self.y = y
        else:
            self.y = int(random.random() * 500)
    
    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'
    
    def distanceTo(self, city):
        xDistance = abs(self.x - city.x)
        yDistance = abs(self.y - city.y)
        return math.sqrt((xDistance * xDistance) + (yDistance * yDistance))


class Tour:
    # can be used as an array containing Cities in travel order
    # takes a list of City objects
    def __init__(self, cityList):
        self.cityList = cityList
        self.fitness = 0.0
        self.distance = 0
        self.tour = [None] * len(self.cityList)
    
    def __len__(self):
        return len(self.tour)
    
    def __getitem__(self, index):
        return self.tour[index]
    
    def __setitem__(self, index, city):
        self.tour[index] = city
        self.fitness = 0.0
        self.distance = 0
    
    def __repr__(self):
        tourString = '-'
        for city in self.tour:
            tourString += str(city) + '-'
        return tourString
    
    # generate a random solution
    def generateIndividual(self):
        for i in range(len(self.cityList)):
            self[i] = self.cityList[i]
        random.shuffle(self.tour)
    
    # fitness score = 1 / travel distance
    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1.0 / float(self.getDistance())
        return self.fitness
    
    def getDistance(self):
        if self.distance == 0:
            for i in range(len(self)):
                src = self[i]
                dest = self[(i+1) % len(self)]
                self.distance += src.distanceTo(dest)
        return self.distance
       
    def contains(self, city):
        return city in self.tour
    
    # show the tour as a graph, requires matplotlib
    def plot(self):
        try:
            imp.find_module('matplotlib')
            import matplotlib.pyplot as plt
            x, y = [], []
            for i in range(len(self)):
                x.append(self[i].x)
                y.append(self[i].y)
            plt.plot(x[0], y[0], 'bs')
            plt.plot(x[1:], y[1:], 'co')
            scale = max(x)/50
            n = len(x)
            for i in range(n):
                plt.arrow(x[i], y[i], (x[(i+1)%n] - x[i]), (y[(i+1)%n] - y[i]),
                          head_width=scale, color='g', length_includes_head=True)
            plt.xlim(0, max(x)*1.1)
            plt.ylim(0, max(y)*1.1)
            plt.show()
        except ImportError:
            print 'Could not find matplotlib'


class Population:
    # can be used as an array containing Tours
    # generate = True to generate an initial population
    def __init__(self, cityList, populationSize, generate):
        self.tours = [None] * populationSize
        if generate:
            for i in range(populationSize):
                newTour = Tour(cityList)
                newTour.generateIndividual()
                self[i] = newTour
    
    def __len__(self):
        return len(self.tours)
    
    def __getitem__(self, index):
        return self.tours[index]
    
    def __setitem__(self, index, tour):
        self.tours[index] = tour
    
    # returns Tour with highest fitness score
    def getFittest(self):
        fittest = self.tours[0]
        for i in range(1, len(self)):
            if fittest.getFitness() < self[i].getFitness():
                fittest = self[i]
        return fittest


class TSPGeneticAlgorithm:
    # forms the GA with a few parameters
    def __init__(self, cityList, populationSize=100, tournamentSize=10, elitism=True, mutationRate=0.02):
        self.cityList = cityList
        self.populationSize = populationSize
        self.tournamentSize = tournamentSize
        self.elitism = elitism
        self.mutationRate = mutationRate
    
    # call this to start the GA
    def run(self, generations=100):
        population = Population(self.cityList, self.populationSize, True)
        print 'Generation#0: ' + str(population.getFittest().getDistance())
        
        for i in range(generations):
            population = self.evolvePopulation(population)
            print 'Generation#' + str(i+1) + ': ' + str(population.getFittest().getDistance())
        
        print 'Solution:'
        print population.getFittest()
        population.getFittest().plot()
    
    # driver code
    def evolvePopulation(self, population):
        evolved = Population(self.cityList, len(population), False)
        elitismOffset = 0
        if self.elitism:
            evolved[0] = population.getFittest()
            elitismOffset = 1
        
        for i in range(elitismOffset, len(evolved), 2):
            parent1 = self.tournamentSelection(population)
            parent2 = self.tournamentSelection(population)
            if i+1 < len(evolved):
                evolved[i], evolved[i+1] = self.crossover(parent1, parent2)
            else:
                evolved[i],_ = self.crossover(parent1, parent2)
        
        for i in range(elitismOffset, len(evolved)):
            self.mutate(evolved[i])
        
        return evolved
    
    # pick n random parents and return the fittest
    def tournamentSelection(self, population):
        tournament = Population(self.cityList, self.tournamentSize, False)
        for i in range(self.tournamentSize):
            randomId = int(random.random() * len(population))
            tournament[i] = population[randomId]
        fittest = tournament.getFittest()
        return fittest
    
    # two point crossover of no more than half the length
    def crossover(self, parent1, parent2):
        pos1 = int(random.random() * len(parent1))
        pos2 = int(random.random() * len(parent1))
        while pos1 == pos2 or abs(pos1 - pos2) > len(parent1)/2:
            pos2 = int(random.random() * len(parent1))
        startPos, endPos = min(pos1, pos2), max(pos1, pos2)
        
        child1 = Tour(self.cityList)
        child1[startPos:endPos+1] = parent2[startPos:endPos+1]
        j = 0
        for i in range(len(parent1)):
            if not child1.contains(parent1[i]):
                while child1[j] != None:
                    j += 1
                child1[j] = parent1[i]
        
        child2 = Tour(self.cityList)
        child2[startPos:endPos+1] = parent1[startPos:endPos+1]
        j = 0
        for i in range(len(parent2)):
            if not child2.contains(parent2[i]):
                while child2[j] != None:
                    j += 1
                child2[j] = parent2[i]
        
        return child1, child2
    
    # mutate by randomly swapping positions in the Tour
    def mutate(self, tour):
        for i in range(len(tour)):
            if random.random() <= self.mutationRate:
                j = int(len(tour) * random.random())
                while j == i:
                    j = int(len(tour) * random.random())
                tour[i], tour[j] = tour[j], tour[i]


if __name__ == '__main__':
    cityList = []
    n = int(raw_input('Enter number of cities: '))
    r = raw_input('Use random city coordinates? (y/n) ')
    for _ in range(n):
        if r == 'y':
            cityList.append(City())
        else:
            x, y = map(int, raw_input('Enter x y: ').split())
            cityList.append(City(x, y))
    
    ga = TSPGeneticAlgorithm(cityList)
    ga.run()
