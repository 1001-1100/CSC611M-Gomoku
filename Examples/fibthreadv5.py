import time
import multiprocessing

from random import randint

class fibthread(multiprocessing.Process):
    def __init__(self, start, total):
        multiprocessing.Process.__init__(self)
        self.total=total
        self.p=start

    def fib(self,input):
        a, b = 0, 1
        for item in range(input):
            a, b = b, a + b
        return a

    def run(self):

        global numlist
        #global lock
        while (self.p < self.total):
            if (self.p<self.total):
                self.fib(numlist[self.p])
                #print("%s \n" %self.fib(numlist[self.p]))
            self.p+=1

if __name__ == '__main__':

    numlist=[]
    element=1000000
    numprocess=10
    inc=int (element/numprocess)
    #print (inc)
    jobs=[]
    #fibobj=[]
    #lock=multiprocessing.Lock()
    #pointer=multiprocessing.Value('i',0)

    print("Generating %s numbers ranging from 0 to 100 into list" % element)
    for i in range(element):
        numlist.append(randint(90,100))

    print ("Number of thread to use: %s"%(numprocess))

    print("Calculating fibonacci numbers")
    start_time=time.time()
    for i in range(numprocess):
        #print(numlist)
        print ("Creating process %s"%i)
        if (i == 0):
            fibobj=fibthread(0,inc)
            #print(i)
            #print ("0 to %s \n"%(inc))
        elif (i == numprocess-1):
            fibobj=fibthread(((inc*i)),element)
            #print (i)
            #print ("%s to %s \n"%((inc*i),element))
        else :
            fibobj=fibthread(((inc*i)), (inc*(i+1)))
            #print(i)
            #print ("%s to %s \n"%((inc*i),((inc*(i+1)))))
        jobs.append(fibobj)
        fibobj.start()

    while len(jobs) > 0:
        jobs=[job for job in jobs if job.is_alive()]

    print("--- %s seconds ---" % (time.time() - start_time))

