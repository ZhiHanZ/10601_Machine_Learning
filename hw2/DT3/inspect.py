import sys
import csv
import math
#readFile(path) and writeFile(path, contents) from http://www.cs.cmu.edu/~112/notes/notes-strings.html

#read from path file and return the contents
def readFile(filePath):
    file = open(filePath)
    csv_file = csv.reader(file)
    return csv_file

#new path file and write content into
def writeFile(filePath, entropy, error):
    with open(filePath, "wt") as file:
        file.write("entropy: %f"%entropy + "\n" 
                 + "error: %f"%error)

def countsDic(allRows):
    countsDic = dict()
 
    for i in range(1, len(allRows)): # 1st row contains features description
        resultIndex = len(allRows[i]) - 1 # the last column is the result
        result = allRows[i][resultIndex]
        countsDic[result] = countsDic.get(result, 0) + 1
    return countsDic

#get path file content reversed to get output file
def inspectData(inputPath, outputPath):
    csv_file = readFile(inputPath)
    allRows = list()

    for row in csv_file:
        allRows.append(row)

    entropy = 0
    counts = countsDic(allRows)
    maxVote = 0
    total = len(allRows) - 1 # the first row are features
    for key in counts.keys():
        if counts[key] > maxVote: maxVote = counts[key]
        p = float(counts[key]) / total 
        entropy += - (p * math.log(p, 2))
    error = 1 - float(maxVote) / total
    writeFile(outputPath, entropy, error)

def main():
    inspectData(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()