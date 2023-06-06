#
# Individual.py
#
#

import math
from random import *
#Base class for all individual types
#
class Individual:
    """
    Individual
    """
    minMutRate=1e-100
    maxMutRate=1
    learningRate=None
    # uniprng=None
    normprng=None
    fitFunc=None

    def __init__(self):
        self.fit=self.__class__.fitFunc(self.state)
        self.mutRate=self.uniprng.uniform(0.9,0.1) #use "normalized" sigma
            
    def mutateMutRate(self):
        self.mutRate=self.mutRate*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.mutRate < self.minMutRate: self.mutRate=self.minMutRate
        if self.mutRate > self.maxMutRate: self.mutRate=self.maxMutRate
            
    def evaluateFitness(self):
        if self.fit == None: self.fit=self.__class__.fitFunc(self.state)


#A combinatorial integer representation class
#
class IntVectorIndividual(Individual):
    """
    IntVectorIndividual
    """
    nLength=None
    # nItems=None
        
    def __init__(self):
        self.state=[]
        for i in range(self.nLength):
            self.state.append(randint(0,1))
        
        super().__init__() #call base class ctor
        
    def crossover(self, other):
        #perform crossover "in-place"
        for i in range(self.nLength):
            if self.uniprng.random() < 0.5:
                tmp=self.state[i]
                self.state[i]=other.state[i]
                other.state[i]=tmp
                        
        self.fit=None
        other.fit=None
    
    def mutate(self):
        self.mutateMutRate() #update mutation rate
        
        for i in range(self.nLength):
            if self.uniprng.random() < self.mutRate:
                if self.state[i] == 0: self.state[i] = 1
                if self.state[i] == 1: self.state[i] = 0

                # self.state[i]= randint(0, 1)
        
        self.fit=None
            
    def __str__(self):
        return str(self.state)+'\t'+'%0.8e'%self.fit+'\t'+'%0.8e'%self.mutRate
    
    
