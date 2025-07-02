import random
import matplotlib.pyplot as plt
from enum import Enum
import time
from statistics import mean
from collections import Counter
from bisect import insort

#global grid variables
GridSize = 10000
maxX = int(GridSize/2)
maxY = maxX
minX = -maxX
minY = minX

#Do we generate the anomaly points on the whole board or outside the intended area 
wholeBoardMode = False

#global variables for search sections
subdivisionModifier = 1
sectionSize = GridSize
sections = []
offsetSections = []
points = []
existingPoints = set()
plots = []


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"


#variables for color cycling
cycleColors = list(Color)
colIndex = 0

#function for generating a point in a gives section
def pointInZone(sector : str):
    retX = retY = 0
    boundaries = { #sector (min X - max X - min Y - max Y)
        "top-left": (-5000, 500, -5000, 500),
        "top-right": (-500, 5000, -5000, 500),
        "bottom-left": (-5000, 500, -500, 5000),
        "bottom-right": (-500, 5000, -500, 5000)
    }
    retX = random.randint(boundaries[sector][0],boundaries[sector][1])
    retY = random.randint(boundaries[sector][2],boundaries[sector][3])

    return retX,retY

#Same as above but generates the point outside the given zone
def pointOutsideZone(sector : str):
    retX = retY = 0
    boundaries = { #sector (min X - max X - min Y - max Y)
        "top-left": (-5000, 500, -5000, 500),
        "top-right": (-500, 5000, -5000, 500),
        "bottom-left": (-5000, 500, -500, 5000),
        "bottom-right": (-500, 5000, -500, 5000)
    }
    retX, retY = random.randint(-5000,5000),random.randint(-5000,5000)
    while((boundaries[sector][0] <= retX <= boundaries[sector][1])or (boundaries[sector][2] <= retY <= boundaries[sector][3])):
        retX, retY = random.randint(-5000,5000),random.randint(-5000,5000)
    

    return retX,retY

#Generates a random coordinate anywhere on the board
def pointOnWholeBoard():
    retX = retY = 0
    retX, retY = random.randint(-5000,5000),random.randint(-5000,5000)
    return retX,retY


def pregenCoord(sector:str):
    #99% chance to generate point in it's proper section
    if(random.randint(1,100)!=1):
        return pointInZone(sector)
    else:
    #1% chance to generate outside that section
        if(wholeBoardMode):
            return pointOnWholeBoard()
        else:
            return pointOutsideZone(sector)



#class for every point on the board; stores color and coordinates 
class point:

    #default constructor
    def __init__(self) -> None:
        global colIndex
        #generate a color according to the currectly circulated color
        self.color = cycleColors[colIndex]
        while(True):
            #generate a coordinate according to the color
            match(self.color):
                case Color.RED:
                    self.x,self.y = pregenCoord("top-left")
                case Color.GREEN:
                    self.x,self.y = pregenCoord("top-right")
                case Color.BLUE:
                    self.x,self.y = pregenCoord("bottom-left")
                case Color.PURPLE:
                    self.x,self.y = pregenCoord("bottom-right")
            #if there is not a point with the same coordinates on the board
            if (self.x, self.y) not in existingPoints:
                #add the point
                existingPoints.add((self.x, self.y))
                colIndex = (colIndex + 1) % len(cycleColors) #and switch to the next color
                break
        
    #alternate constructor for preset points
    @classmethod
    def with_coordinates(cls,x: int, y: int, color: Color):
        obj = cls()
        obj.x, obj.y = x, y
        existingPoints.add((x, y))
        xSect,ySect = getSection(obj.x,obj.y)
        sections[xSect][ySect].append(obj)
        if pointWithinOffset(obj.x,obj.y):
            x_index_off, y_index_off = getOffsetSection(obj.x,obj.y)
            offsetSections[x_index_off][y_index_off].append(obj)
        obj.color = color
        return obj
        

#manhattan distance used for faster computation speed
def manhattanDistance(x1: int, y1:int, x2 : int, y2 : int):
    return abs(x2 - x1) + abs(y1 - y2)


def classify(x : int, y : int, k : int) -> Color:
    closestPoints = []
    sectX, sectY = getSection(x,y)                  #find the section to which the new coordinate belongs
    sectionToSearch = set(sections[sectX][sectY])   #and add the section to the searched zone
    if pointWithinOffset(x,y):                      #check if the point belongs in an offset section
        xOff, yOff = getOffsetSection(x,y)          #and add the offset section to search area if it is
        sectionToSearch = ((sectionToSearch) | set(offsetSections[xOff][yOff]))

    for point in sectionToSearch:                                       #go through every point in the search area
        distance = manhattanDistance(x, y, point.x, point.y)            #calculate distance from each point
        insort(closestPoints, (distance, point), key=lambda x: x[0])    #insert the point into an array in such a way that it is sorted
        if len(closestPoints) > k:
            closestPoints.pop()                                         #remove the furthermost point if we have enough points for k


    #if we did not find enough points in the initial search, go through the same search process    
    if len(closestPoints) < k:  #but this time check the entire board
        closestPoints.clear()   #with a clean points array to avoid duplicates
        for point in points:
            distance = manhattanDistance(x, y, point.x, point.y)
            insort(closestPoints, (distance, point), key=lambda x: x[0])
            if len(closestPoints) > k:
                closestPoints.pop()

    colors = [point[1].color for point in closestPoints]        #find the most common color among k closest points

    mostCommonColor = Counter(colors).most_common(1)[0][0]
    return mostCommonColor

#find in which section a given point is
def getSection(x : int, y : int):
    x_index = min(int((x - minX) / sectionSize), subdivisionModifier - 1)
    y_index = min(int((y - minY) / sectionSize), subdivisionModifier - 1)
    
    return x_index, y_index

#same as above function buut for offset section
def getOffsetSection(x : int, y : int):
    x_index = min(int((x - (minX + sectionSize / 2)) / sectionSize), subdivisionModifier - 2)
    y_index = min(int((y - (minY + sectionSize / 2)) / sectionSize), subdivisionModifier - 2)
    
    return x_index, y_index

#check if the point is in offset area
def pointWithinOffset(x,y):
    return minX + sectionSize / 2 <= x <= maxX - sectionSize / 2 and minY + sectionSize / 2 <= y <= maxY - sectionSize / 2


def main(subdivide : int = 49, k = 1,showResults = True, seed = None):
    global sections, sectionSize, subdivisionModifier, points, offsetSections
    
    #if we provide a seed to the function, use the seed
    if(seed is not None):
        random.seed(seed)
    print(f"starting the simulation for {subdivide} subdivisions and k={k}")
    subdivisionModifier = subdivide

    #according to the given subdivide value, calculate section size and initiate section arrays
    sectionSize = GridSize / subdivisionModifier
    sections = [[[] for _ in range(subdivisionModifier)] for _ in range(subdivisionModifier)]
    offsetSections = [[[] for _ in range(subdivisionModifier - 1)] for _ in range(subdivisionModifier - 1)]

    #pregen the initial points
    R = [point.with_coordinates(-4500, -4400,Color.RED), point.with_coordinates(-4100, -3000,Color.RED), point.with_coordinates(-1800, -2400,Color.RED), point.with_coordinates(-2500, -3400,Color.RED) , point.with_coordinates(-2000, -1400,Color.RED)]
    G = [point.with_coordinates(+4500, -4400,Color.GREEN), point.with_coordinates(+4100, -3000,Color.GREEN), point.with_coordinates(+1800, -2400,Color.GREEN), point.with_coordinates(+2500, -3400,Color.GREEN) , point.with_coordinates(+2000, -1400,Color.GREEN)]
    B = [point.with_coordinates(-4500, +4400,Color.BLUE), point.with_coordinates(-4100, +3000,Color.BLUE), point.with_coordinates(-1800, +2400,Color.BLUE), point.with_coordinates(-2500, +3400,Color.BLUE) ,point.with_coordinates(-2000, +1400,Color.BLUE)]
    P = [point.with_coordinates(+4500, +4400,Color.PURPLE), point.with_coordinates(+4100, +3000,Color.PURPLE), point.with_coordinates(+1800, +2400,Color.PURPLE), point.with_coordinates(+2500, +3400,Color.PURPLE) , point.with_coordinates(+2000, +1400,Color.PURPLE)]
    points = R+G+B+P



    startTime = time.time()
    pointCount = 40000
    goodPointCount = 0
    percentageProgress = 0
    for i in range(pointCount): #generate our points
        percentageProgress = (i+1)/pointCount*100
        if(percentageProgress%5 == 0):                      #this is just for monitoring the progress
            print(f"{percentageProgress}% done",end=' ')
            
        newpoint = point()                                  #generate a point with a color and coordinates according to the color
        newCol = classify(newpoint.x,newpoint.y,k)          #classify the point
        newpoint.color = newCol                             #assign the new color
        points.append(newpoint)                             
        xSect,ySect = getSection(newpoint.x,newpoint.y)     #add the point to the global points array, a main section and, if applicable, an offset section
        sections[xSect][ySect].append(newpoint)
        if pointWithinOffset(newpoint.x,newpoint.y):
            x_index_off, y_index_off = getOffsetSection(newpoint.x,newpoint.y)
            offsetSections[x_index_off][y_index_off].append(newpoint)
        
        #check if the point was placed in it's intended section, increment good point count if it is
        match(newCol):
            case Color.RED:
                goodPointCount+=(newpoint.x < 500 and newpoint.y < 500)
            case Color.GREEN:
                goodPointCount+=(newpoint.x > -500 and newpoint.y < 500)
            case Color.BLUE:
                goodPointCount+=(newpoint.x < 500 and newpoint.y > -500)
            case Color.PURPLE:
                goodPointCount+=(newpoint.x > -500 and newpoint.y > -500)

    #calculate time and success rate
    timeTaken = time.time()-startTime
    print(f"\nTime taken: {timeTaken}")
    successRate = (goodPointCount/pointCount)*100 #success rate is calculated based on how many points were placed in their sections
    print(f"Success rate: {successRate}%")
    
    
    if(showResults): #generate and show the graph
        x_values = [point.x for point in points]
        y_values = [-point.y for point in points]
        colors = [point.color.value for point in points]
        plt.figure(figsize=(8, 8))
        plt.scatter(x_values, y_values, c=colors, s=20)
        plt.show()
        plots.append((x_values, y_values, colors, f"Plot for k={k}"))
        plt.close('all')

    #reset the global arrays
    points.clear()
    sections.clear()
    offsetSections.clear()
    existingPoints.clear()
    print("\n")
    return successRate,timeTaken




if __name__=="__main__":
    action = input("Would you like to generate anomaly points on the whole board or outside their intended area?\n['w' for whole, 'o' for outside their zones]")
    while(True):
        if(action=='w'):
            wholeBoardMode=True
            print("Whole Board mode selected")
            break
        elif(action == 'o'):
            wholeBoardMode=False
            print("Generating anomaly points outside their intended zone")
            break

    action = input("Would you like to use the demonstration mode(DEFAULT) or testing?\n['t' for testing, otherwise chooses the default, intended way of viewing the program]")
    while(True):
        if(action!='t'):
            #generate a seed to use every time we call main
            seed = random.randrange(9999999999)
            main(k=1, subdivide=49,seed=seed)
            main(k=3, subdivide=40,seed=seed)
            main(k=7, subdivide=20,seed=seed)
            main(k=15, subdivide=15,seed=seed)
            
            #this block generates the graphs for the 4 times the program was run above; partly generated by ai and modified by myself for my purposes
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))
            for idx, (x, y, colors, title) in enumerate(plots):
                row, col = divmod(idx, 2)
                axs[row, col].scatter(x, y, c=colors, s=20)
                axs[row, col].set_title(title)
                axs[row, col].set_xlabel("X")
                axs[row, col].set_ylabel("Y")
                axs[row, col].grid(True)

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
            plt.show()
            break
        else:
            performances = []
            for i in [(1,49),(3,40),(7,20),(15,15)]:
                resultsForSubdivision = []
                for j in range(100):
                    currRestult = main(k=i[0],subdivide=i[1],showResults=False)
                    resultsForSubdivision.append(currRestult[0])
                performances.append((i,mean(resultsForSubdivision)))
                print(f"Simulation for {i} subdivisions was finished\n")
            print(performances)
            for p in performances:
                print(p)
            bestPerf = max(performances, key=lambda x: x[0])
            print(f"Best performance: {bestPerf}")


            x_vals = [str(p[0][0]) for p in performances]
            y_vals = [(p[1]) for p in performances]

            plt.bar(x_vals, y_vals, color='skyblue')
            plt.ylim(60, 100) 


            plt.xlabel('k')
            plt.ylabel('Performance Score')
            plt.title('Performance vs k value')

            plt.show()
            break

    #this block runs the default simulations with the calculated best subdivide(section count)
    
    
    #this block was used for testing the different k values for their success rates in it's current form;
    # can be modified to test other values, as has been done for optimal subdivide values.
    # This tester tests the generator with random seeds, to see the success of the function on average, independantly of the seed.
    