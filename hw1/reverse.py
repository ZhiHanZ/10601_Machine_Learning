import sys
#readFile(path) and writeFile(path, contents) from http://www.cs.cmu.edu/~112/notes/notes-strings.html

#read from path file and return the contents
def readFile(path):
    with open(path, "rt") as file:
        return file.read()

#new path file and write content into
def writeFile(path, contents):
    with open(path, "wt") as file:
        file.write(contents)

#get path file content reversed to get output file
def reverseFile(inputPath, outputPath):
    contents = readFile(inputPath)
    newContents = ""
    wordList = contents.split("\n")
    wordList.pop() #remove empty string
    wordList.reverse()
    for word in wordList:
        newContents += word + "\n"
    writeFile(outputPath, newContents)

def main():
    reverseFile(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()