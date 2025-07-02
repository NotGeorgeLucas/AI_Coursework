import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt


#THIS PROGRAM USES A LOT OF TYPE: IGNORE STATEMENTS, DUE TO A PROBLEM WITH MY IDE. PLEASE IGNORE THEM.
# THESE DO NOT AFFECT THE CODE IN ANY WAY, AND ONLY SUPPRESS WARNINGS THAT MY IDE GIVES.

#These are global variables that can be altered to change the parameters for the algorithm
startX = 3
startY = 6
treasureMaxCount = 5
maxX = 7
maxY = 7
mutationPercent = 1
maxMutationPercent = 15
populationCount = 200



#The class for instruction cells; saves the cell information as a string
class Instruction:
    def __init__(self, number):
        self.containedInfo = format(number, '08b') #I found out about this function through ChatGPT, but later used it myself

    def __repr__(self):
        return self.containedInfo
    
    #increment the value in the cell; if reached past 11111111 loop around to 00000000
    def increment(self):
        self.containedInfo = format(int(self.containedInfo,2)+1,'08b')
        if(int(self.containedInfo,2)>255): #I found out about this function through ChatGPT, but later used it myself
            self.containedInfo = format(0,'08b')
    
    #decrement the value in the cell; if reached past 00000000 loop around to 11111111
    def decrement(self):
        self.containedInfo = format(int(self.containedInfo,2)-1,'08b')
        if(int(self.containedInfo,2)<0):
            self.containedInfo = format(255,'08b')

    def getInstruction(self):
        return self.containedInfo
    
    #decyphers the instruction stored in the cell according to the rules in the documentation;
    #returns the instruction and the adress of the cell that the instruction is targeted at
    def readInstruction(self):
        action = ""
        match(self.containedInfo[:2]):
            case '00':
                action = "increment"
            case '01':
                action = "decrement"
            case '10':
                action = "jump"
            case '11':
                action = "output"

        adress = self.containedInfo[2:]

        return (action,adress)

#converts the adress in the cell to a direction, according to the last two positions in the cell,
#00 correlates to a RIGHT move, 01 correlates to an UP move, 10 correlates to an LEFT move, 11 correlates to an DOWN move 
def binToDir(bin):
    match bin[-2:]:
        case '00':
            return 'P'
        case '01':
            return 'H'
        case '10':
            return 'L'
        case '11':
            return 'D'


#simulates the instructions and executes them on a board
def runVM(inputInstructions,printPath = False):   

    instructions = inputInstructions.copy()

    currentInstruction = 0
    instructionCount = 0
    moveCount = 0
    wrongMoves = 0

    treasureCount = 0
    currX, currY = startX, startY
    fieldArray = [[0 for i in range(maxX)] for j in range(maxY)]


    

    treasures =  [(1,4),(2,2),(3,6),(4,1),(5,4)]
    for i in treasures:
        y, x = i
        fieldArray[y][x] = 1
    fieldArray[currY][currX] = 'S' # type: ignore
    #the above sequence of commands sets up the board with the player position, treasure position and boundaries

    if(printPath):
        for i in fieldArray:
            print(i)
        print("\n")
        time.sleep(0.5)
    while True:
        jumped = False
        instructionCount+=1
        (currentAction,adress) = instructions[currentInstruction].readInstruction() #this line extracts the instruction from the cell
        move = None
        match(currentAction): #this case statement processes the instruction
            case "increment":
                instructions[int(adress,2)].increment()
                if(printPath):
                    print(f"incrementing {adress}")
                    time.sleep(0.15)
            case "decrement":
                instructions[int(adress,2)].decrement()
                if(printPath):
                    print(f"decrementing {adress}")
                    time.sleep(0.15)
            case "jump":
                currentInstruction = int(adress,2)
                jumped = True
                if(printPath):
                    print(f"jumping to {adress}")
                    time.sleep(0.15)
            case "output":
                move = binToDir(instructions[int(adress,2)].getInstruction())
                moveCount+=1


        if(move != None): #if we moved, figure out if we can move and try to move in that direction
            match(move):
                case 'D':
                    if(currY==maxY-1):
                        wrongMoves+=1
                        break
                    else:
                        fieldArray[currY][currX] = 0
                        currY+=1
                        fieldArray[currY][currX] = 'S' # type: ignore                        
                case 'H':
                    if(currY==0):
                        wrongMoves+=1
                        break
                    else:
                        fieldArray[currY][currX] = 0
                        currY-=1
                        fieldArray[currY][currX] = 'S' # type: ignore
                case 'L':
                    if(currX==0):
                        wrongMoves+=1
                        break
                    else:
                        fieldArray[currY][currX] = 0
                        currX-=1
                        fieldArray[currY][currX] = 'S' # type: ignore
                case 'P':
                    if(currX==maxX-1):
                        wrongMoves+=1
                        break
                    else:
                        fieldArray[currY][currX] = 0
                        currX+=1
                        fieldArray[currY][currX] = 'S' # type: ignore

            if((currY,currX) in treasures): #collect the treasure if we stepped on one
                treasures.remove((currY,currX))
                treasureCount+=1
            if(printPath):
                for i in fieldArray:
                    print(i)
                print("\n")
                time.sleep(0.7)


        if(not jumped):
            currentInstruction+=1
        if(instructionCount>500):
            break
        if(treasureCount == treasureMaxCount):
            break
        if(currentInstruction>63):
            break

    return (treasureCount,moveCount)



#class that stores the chromosome, it's genes(instructions) and it's statistics after the simulation
class Chromosome:
    def __init__(self,instructions):
        self.instructions = instructions
        self.treasureCount = 0
        self.moveCount = 0
        self.fitness = 0

    def setValues(self, treasureCount, moveCount):
        self.treasureCount = treasureCount
        self.moveCount = moveCount
        self.fitness = calculateFitness(treasureCount,moveCount)

    def getFitness(self):
        return self.fitness        

    def __repr__(self):
        return f"{self.instructions} {self.fitness} WITH {self.treasureCount} TREASURES {self.moveCount} MOVES."


#FITNESS FUNCTION that primarily values the increase in treasures found 
# and secondarily the amount of moves needed to reach that result
def calculateFitness(treasureCount, moveCount):
    return int(treasureCount * 500) - moveCount 




def tournamentChoice(population): #The BASE for this FUNCTION has been written by AI and later altered by me 
    k = int(max(len(population)*0.02,3))
    battlers = random.sample(population ,k=k)
    return max(battlers,key=lambda x: x.fitness)

#crossover function, that, based on the method selected at the start, crosses over the chromosomes
# using a randomly generated mask, merging two chromosomes selected for breeding; calls the mutate function at the end 
# ranking method has a chance of using the genes of the global fittest chormosomes in the process
def crossOver(population):
    if(method == "t"):
        selectedPercent = 0
        newPop = []
    else:
        selectedPercent = 0.15
        newPop = population[:int(len(population)*selectedPercent)]
    choicePop = population[:int(len(population)*selectedPercent)]
    choicePop.append(fittestChromosome)
    while(len(newPop)<populationCount):
        if method == "t":
            parents = [tournamentChoice(population),tournamentChoice(population)]
        else:
            parents = [random.choice(choicePop), random.choice(choicePop)]


        mask1 = [random.randint(0, 1) for _ in range(len(parents[0].instructions))]

        childMoves = []
        for i,maskPar in enumerate(mask1):
            childMoves.append(parents[maskPar].instructions[i])
        newPop.append(Chromosome(childMoves))

    global mutationPercent, maxMutationPercent
    mutationPercent = min(mutationPercent+1,maxMutationPercent)
    
    return newPop[:int(len(population)*selectedPercent)]+mutate(newPop[int(len(population)*selectedPercent):])


#mutates 2 or 3 genes in a chromosomes with a certain chance,
# flipping a bit in each gene, and, with a 10% chance, flipping another bit in that same gene
def mutate(population):
    global mutationPercent

    for chromosome in population:
        if random.randrange(100) <= mutationPercent:
            mutationCount = random.randint(2,3)
            genesToMutate = random.sample(chromosome.instructions,k=mutationCount) #This SPECIFIC line has been written by ChatGPT
            for i, geneToMod in enumerate(genesToMutate):
                gene = list(geneToMod.containedInfo)
                index_to_flip = random.randrange(len(gene))
                gene[index_to_flip] = '1' if gene[index_to_flip] == '0' else '0' #This SPECIFIC line has been written by ChatGPT
                if(random.randrange(100)>90):
                    index_to_flip = random.randrange(len(gene))
                    gene[index_to_flip] = '1' if gene[index_to_flip] == '0' else '0' #This SPECIFIC line has been written by ChatGPT
                chromosome.instructions[i].containedInfo = ''.join(gene)

    return population


#creates a completely random starter population
def createPopoulation():
    global populationCount
    population = [0]* populationCount    
    for i in range(populationCount):
        instructions = [Instruction(random.randrange(255)) for _ in range(64)]

        population[i] = Chromosome(instructions)  # type: ignore
    return population



method = "a"
fittestChromosome = Chromosome([Instruction(0)]*64)
def main():
    global mutationPercent,method,fittestChromosome
    shouldRun = True

    while(True):
        action = input("type r if you would like to use ranking method; type t if you would like to use tournament method ")
        if(action == "r" or action == "t"):
            method = action
            break

    generationIndex = 0

    generationsWithoutImprovement = 0

    population = createPopoulation()
    averageList = []
    bestList = []
    
    fittestChromosome = Chromosome([Instruction(0)]*64)
    fittestChromosome.setValues(0,500)

    attemptNumber = 1
    
    while shouldRun:
        generationIndex +=1
        #simulate the interactions of every chromosome with the board
        for i,chromosome in enumerate(population):
            instructions = deepcopy(chromosome.instructions) # type: ignore
            (treasureCount, moveCount) = runVM(instructions)

            chromosome.setValues(treasureCount,moveCount) # type: ignore
        population = sorted(population, key=lambda x: -x.fitness) # type: ignore #This SPECIFIC line has been written by ChatGPT

        #calculate the best and average results for a generation
        averageList.append(sum(chr.fitness for chr in population)/len(population)) # type: ignore
        bestList.append(population[0].fitness) # type: ignore

        #if a new best chromosome was found, save that result
        if fittestChromosome.fitness < population[0].fitness: # type: ignore
            fittestChromosome = Chromosome(population[0].instructions) # type: ignore
            fittestChromosome.setValues(population[0].treasureCount,population[0].moveCount) # type: ignore
            generationsWithoutImprovement = 0
        else:
            generationsWithoutImprovement+=1

        print(f'Best In Gen Number {generationIndex}: {population[0]}\n')
        print(f"Best Result: {fittestChromosome}\n")
        #console interaction code for when 5 treasres were found
        if fittestChromosome.treasureCount == treasureMaxCount:
            while shouldRun:
                action = input(f"After {attemptNumber} attempt(s), a path to {treasureMaxCount} treasure(s) was found; would you like to see the path?(y/n) ")
                if action == "y":
                    instructions = deepcopy(fittestChromosome.instructions) # type: ignore
                    runVM(instructions,printPath=True)
                    shouldRun = False
                elif action == "n":
                    action = input("type y if you would like to simulate another generation; else, the program will close ")
                    if action !="y":
                        shouldRun = False
                    else:
                        break
        #if we didn't improve our result in a long time, restart the simlation
        if generationsWithoutImprovement == 50:
            print(f"{generationsWithoutImprovement} generations have passed without finding {treasureMaxCount} treasures or improving current best have been reached;\nRestarting algorithm")
            time.sleep(3)
            fittestChromosome = Chromosome([Instruction(0)]*64)
            fittestChromosome.setValues(0,500)
            population = createPopoulation()
            mutationPercent=1    
            generationIndex = 0
            averageList.clear()
            bestList.clear()
            attemptNumber+=1
            continue
        #if we should continue, create a new population and restart the loop
        population = crossOver(population)
    
    
    #The part of the code responsible for drawing the graph was written by ChatGPT
    plt.plot(averageList, label="Average", marker='o')
    plt.plot(bestList, label="Best", marker='x')
    
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Generation progress graph')
    
    plt.legend()

    plt.show()
    

            



if __name__ == "__main__":
    
    main()