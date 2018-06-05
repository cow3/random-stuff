from text import *
def samenames(a,b):
    return "".join(a.split(" ")).upper()=="".join(b.split(" ")).upper()
class anime:
    name = None
    Kscore = None
    Complete = None
    def __init__(self,info):
        self.Complete, self.name, self.Kscore = info.split("\t")
        self.Complete=bool(int(self.Complete))
        self.Kscore=float(self.Kscore)

    def __repr__(self):
        return self.name

    def info(self):
        return self.position() + ".Name: " + self.name[:50] + " "*(50-len(self.name)) +"Completed: " + str(self.Complete) + " "*(2+self.Complete) +"Kitsu score: " + str(self.Kscore) + " "*(6-len(str(self.Kscore))) + "Scaled score: " + str(self.scaled_score())

    def position(self):
        count=1
        for a in anilist:
            assert(type(a)==anime)
            if a.Kscore > self.Kscore:
                count+=1
        return "0"*(3-len(str(count))) + str(count)

    def scaled_score(self):
        count=0
        for a in anilist:
            assert(type(a)==anime)
            if a.Kscore > self.Kscore:
                count+=1
        return 10- (10*count)/(len(anilist)-1)

    def __iter__(self):
        return self

anilist = fileToList("Anime List.txt")
for i in range(len(anilist)):
    anilist[i]=anime(anilist[i])

anilist.sort(key=lambda x:x.Kscore)

continuing = True
while continuing:
    print("please enter one of the following numbers:")
    print("1. add anime")
    print("2. remove anime")
    print("3. view specific anime")
    print("4. view anime list")
    print("5. general information")
    print("6. backup anime list")
    print("7. quit/exit")
    I = input()
    if I == "1":
        print("please enter new anime name")
        I2= input()+"\t"
        print("please enter if anime has been completed 0/1")
        I3= input()+"\t"
        print("please enter kscore")
        I4= input()
        newanime=anime(I3+I2+I4)
        anilist.append(newanime)
        anilist.sort(key=lambda x:x.Kscore)
        listToFile("Anime List.txt",[str(int(x.Complete)) +"\t" + x.name +"\t" + str(x.Kscore) for x in anilist])
        print("the following anime has been added to your list:")
        print(newanime.info())
    if I == "2":
        print("please enter name of anime to remove")
        I2=input()
        for a in anilist:
            if samenames(a,I2):
                anilist.remove(a)
                break
        listToFile("Anime List.txt",[str(int(x.Complete)) +"\t" + x.name +"\t" + str(x.Kscore) for x in anilist])
        print("the following anime has been removed from your list:")
        print(a.info())
    if I == "3":
        print("please enter name of anime")
        I2=input()
        for a in anilist:
            if samenames(a.name,I2):
                 print(a.info())
    if I == "4":
        for a in anilist:
            print(a.info())
    if I == "5":
        count = 0
        for a in anilist:
            count += a.Complete
        print("you have completed "+ str(count) +" anime")
    if I == "6":
        listToFile("animelist backup.txt",[str(int(x.Complete)) +"\t" + x.name +"\t" + str(x.Kscore) for x in anilist])
        print("your anime list has been backed up")
    if I == "7":
        continuing = False
    I=""
