#
# ev3a.py: An elitist (mu+mu) generational-with-overlap EA
#
#
# To run: python ev3a.py --input ev3a_example.cfg
#         python ev3a.py --input my_params.cfg
#
# Basic features of ev3a:
#   - Supports self-adaptive mutation
#   - Uses binary tournament selection for mating pool
#   - Uses elitist truncation selection for survivors
#   - Supports IntegerVector and Multivariate Individual types
#



import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


import optparse
import sys
import yaml
import math
from random import Random
from Population import *
from Evaluator import *
from random import *

#EV3 Config class 
class EV3_Config:
    """
    EV3 configuration class
    """
    # class variables
    sectionName='EV3'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'evaluator': (str,True),
             'n_features': (int,False),
             'minLimit': (float,False),
             'maxLimit': (float,False),
             'path': (str, True)}
     
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV3 section
        infile=open(inFileName,'r')
        ymlcfg=yaml.safe_load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))
         
        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]
 
                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                 
                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
     
    #string representation for class data    
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))
         


#Print some useful stats to screen
def printStats(pop,gen):
    print('Generation:',gen)
    avgval=0
    maxval=pop[0].fit 
    mutRate=pop[0].mutRate
    for ind in pop:
        avgval+=ind.fit
        if ind.fit > maxval:
            maxval=ind.fit
            mutRate=ind.mutRate
        print(ind)

    print('Max fitness',maxval)
    print('MutRate',mutRate)
    print('Avg fitness',avgval/len(pop))
    print('')


#EV3:
#            
def ev3(cfg):
    #start random number generators
    uniprng=Random()
    uniprng.seed(cfg.randomSeed)
    normprng=Random()
    normprng.seed(cfg.randomSeed+101)

    #set static params on classes
    Features_selection.path = cfg.path
    Individual.uniprng=uniprng
    Individual.normprng=normprng
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg.crossoverFraction
    IntVectorIndividual.fitFunc=Features_selection.fitnessFunc
    IntVectorIndividual.nLength=cfg.n_features
    IntVectorIndividual.learningRate=1.0/math.sqrt(cfg.n_features)
    Population.individualType=IntVectorIndividual
    
    
    
    
    #create initial Population (random initialization)
    population=Population(cfg.populationSize)
        
    #print initial pop stats    
    printStats(population,0)

    #evolution main loop
    for i in range(cfg.generationCount):
        #create initial offspring population by copying parent pop
        offspring=population.copy()
        
        #select mating pool
        offspring.conductTournament()

        #perform crossover
        offspring.crossover()
        
        #random mutation
        offspring.mutate()
        
        #update fitness values
        offspring.evaluateFitness()        
            
        #survivor selection: elitist truncation using parents+offspring
        population.combinePops(offspring)
        population.truncateSelect(cfg.populationSize)
        
        #print population stats    
        printStats(population,i+1)

    best_Ind = population[-1].state
    
    
    # get features subset
    features_col = ['c0','c1','c2','c3','c4','c5','c6','c7','c8']
    cols = []
    n = 0
    for i in range(len(best_Ind)):
        if best_Ind[i] != 0:
            cols.append(features_col[i])
            n+=1

    print("the best combination of features is: ", best_Ind)
    print("The number of fearures in the Best subset is :", n)
    print("the Best subset of features is composed as follow: " , cols)

        





# Main entry point
#
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)
        
        #validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")
        
        #Get EV3 config params
        cfg=EV3_Config(options.inputFileName)
        
        #print config params
        print(cfg)
                    
        #run EV3
        ev3(cfg)
        
        if not options.quietMode:                    
            print('====================COMPLETED====================')    
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)
    

if __name__ == '__main__':
    main()
    
