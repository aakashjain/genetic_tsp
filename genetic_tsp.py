import math
import random
import sys
import imp


class City:
    def __init__(self, x=None, y=None):
        self.x = None
        self.y = None
        if x is not None:
            self.x = x
        else:
            self.x = int(random.random() * 200)
        if y is not None:
            self.y = y
        else:
            self.y = int(random.random() * 200)
    
    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'
    
    def distanceTo(self, city):
        xDistance = abs(self.x - city.x)
        yDistance = abs(self.y - city.y)
        return math.sqrt((xDistance * xDistance) + (yDistance * yDistance))


class TourManager:
    def __init__(self):
        self.destinationCities = []
    
    def __getitem__(self, index):
        return self.destinationCities[index]

    def __len__(self):
        return len(self.destinationCities)

    def append(self, city):
        self.destinationCities.append(city)
    
    def length(self):
        return len(self.destinationCities)


class Tour:
    def __init__(self, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.fitness = 0.0
        self.distance = 0
        if tour is not None:
            self.tour = tour
        else:
            self.tour = [None] * len(self.tourmanager)
    
    def __len__(self):
        return len(self.tour)
    
    def __getitem__(self, index):
        return self.tour[index]
    
    def __setitem__(self, index, city):
        self.tour[index] = city
        self.fitness = 0.0
        self.distance = 0
    
    def __repr__(self):
        tourString = '|'
        for city in self.tour:
            tourString += str(city) + '|'
        return tourString
    
    def generateIndividual(self):
        for i in range(len(self.tourmanager)):
            self[i] = self.tourmanager[i]
        random.shuffle(self.tour)
    
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
            n = len(x)
            for i in range(n):
                plt.arrow(x[i], y[i], (x[(i+1)%n] - x[i]), (y[(i+1)%n] - y[i]),
                          head_width=4, color='g', length_includes_head=True)
            plt.xlim(0, max(x)*1.1)
            plt.ylim(0, max(y)*1.1)
            plt.show()
        except ImportError:
            print 'Could not find matplotlib'


class Population:
    def __init__(self, tourmanager, populationSize, generate):
        self.tours = [None] * populationSize
        if generate:
            for i in range(populationSize):
                newTour = Tour(tourmanager)
                newTour.generateIndividual()
                self[i] = newTour
    
    def __len__(self):
        return len(self.tours)

    def __getitem__(self, index):
        return self.tours[index]
    
    def __setitem__(self, index, tour):
        self.tours[index] = tour
    
    def getFittest(self):
        fittest = self.tours[0]
        for i in range(1, len(self)):
            if fittest.getFitness() <= self[i].getFitness():
                fittest = self[i]
        return fittest


class TSPGeneticAlgorithm:
    def __init__(self, tourmanager, mutationRate=0.015, tournamentSize=5, elitism=True):
        self.tourmanager = tourmanager
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.elitism = elitism
    
    def evolvePopulation(self, population):
        evolved = Population(self.tourmanager, len(population), False)
        elitismOffset = 0
        if self.elitism:
            evolved[0] = population.getFittest()
            elitismOffset = 1
        
        for i in range(elitismOffset, len(evolved)):
            parent1 = self.tournamentSelection(population)
            parent2 = self.tournamentSelection(population)
            evolved[i] = self.crossover(parent1, parent2)
        
        for i in range(elitismOffset, len(evolved)):
            self.mutate(evolved[i])
        
        return evolved
    
    def tournamentSelection(self, population):
        tournament = Population(self.tourmanager, self.tournamentSize, False)
        for i in range(self.tournamentSize):
            randomId = int(random.random() * len(population))
            tournament[i] = population[randomId]
        fittest = tournament.getFittest()
        return fittest

    def crossover(self, parent1, parent2):
        child = Tour(self.tourmanager)

        pos1 = int(random.random() * len(parent1))
        pos2 = int(random.random() * len(parent2))
        startPos, endPos = min(pos1, pos2), max(pos1, pos2)
        for i in range(startPos, endPos+1):
            child[i] = parent1[i]
        
        for i in range(len(parent2)):
            if not child.contains(parent2[i]):
                for j in range(len(child)):
                    if child[j] == None:
                        child[j] = parent2[i]
                        break
        return child
    
    def mutate(self, tour):
        for i in range(len(tour)):
            if random.random() <= self.mutationRate:
                j = int(len(tour) * random.random())
                tour[i], tour[j] = tour[j], tour[i]


if __name__ == '__main__':
    tourmanager = TourManager()
    print 'Enter number of cities: '
    nCities = int(raw_input())
    print '(e)nter coordinates or use (r)andom cities? '
    r = raw_input()
    for _ in range(nCities):
        if r == 'r':
            tourmanager.append(City())
        elif r == 'e':
            print 'Enter x y: '
            x, y = map(int, raw_input().split())
            tourmanager.append(City(x, y))
        else:
            print 'Invalid'
            sys.exit(0)
    
    population = Population(tourmanager, 50, True);
    print 'Generation#0: ' + str(population.getFittest().getDistance())
    
    ga = TSPGeneticAlgorithm(tourmanager)
    for i in range(100):
        population = ga.evolvePopulation(population)
        print 'Generation#' + str(i+1) + ': ' + str(population.getFittest().getDistance())
    
    print 'Solution:'
    print population.getFittest()
    population.getFittest().plot()