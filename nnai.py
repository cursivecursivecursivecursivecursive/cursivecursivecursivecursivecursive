import pygame
import random
import math
import numpy as np
import string

#----------FUNCTIONS

def distance(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    return math.sqrt((dx * dx) + (dy * dy))

def calc_point(ox, oy, oa, a, r):

        newAngle = oa + a

        if newAngle > 359:
            newAngle -= 359

        x = ox + r * math.cos(newAngle * math.pi / 180)
        y = oy + r * math.sin(newAngle * math.pi / 180)
        return x, y

def getAngleBetweenPoints(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    y = y2 - y1
    x = x2 - x1

    return (math.atan2(y, x) * 180 / math.pi)

#----------ARTIFACTS

class simulation:
    def __init__(self):
        pass

#simple loss
class loss:
    def calc(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

#loss by categorical cross-entropy
class l_cce(loss):
    def forward(self, y_pred, y_tar):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #scalar
        if len(y_tar.shape) == 1:
            correct_conf = y_pred_clipped[range(samples), y_tar]
        #1hot
        elif len(y_tar.shape) == 2:
            correct_conf = np.sum(y_pred_clipped * y_tar, axis = 1)

        return -np.log(correct_conf)

#rectified linear activation
class ac_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#softmax activation
class ac_sm:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probables = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probables

#----------CLASSES

class neural_network:

    def __init__(self, org, i, h, o):

        self.organism = None
        self.layers = []
        self.inputLayer = None
        self.outputLayer = None

        self.organism = org

        self.inputLayer = layer(i)
        self.outputLayer = layer(o)

        self.layers.append(self.inputLayer)

        for l in range(h[0]):

            hiddenLayer = layer(h[1])
            self.layers.append(hiddenLayer)

        self.layers.append(self.outputLayer)

        previousLayer = None

        for l in self.layers:
            if previousLayer != None:
                for n in previousLayer.getNeurons():
                    for nn in l.getNeurons():
                        weight = random.randrange(0, 100)
                        newSynapse = synapse((weight/100))

                        n.addOutput(newSynapse)
                        newSynapse.addOutput(nn)
            else:
                previousLayer = l

    def getLayers(self):
        return self.layers

    def getOrganism(self):
        return self.organism

    def forward(self, inputs):
        for n in range(len(inputs)):
            self.inputLayer.getNeurons()[n].setValue(inputs[n])

        for l in self.layers:
            l.forward()

        highestValue = 0
        bestNeuron = -1
        for n in range(len(self.outputLayer.getNeurons())):
            if self.outputLayer.getNeurons()[n].getValue() > highestValue:
                highestValue = self.outputLayer.getNeurons()[n].getValue()
                bestNeuron = n

        match bestNeuron:
            case 0:
                self.organism.rotate(-1)
            case 1:
                self.organism.move(1)
            case 2:
                self.organism.rotate(1)

        for l in self.layers:
            for n in l.getNeurons():
                n.setValue(0)

class layer:

    def __init__(self, n):

        self.neurons = []

        for nn in range(n):
            bias = random.randrange(0, 100)
            newNeuron = neuron((bias/100))

            self.neurons.append(newNeuron)

    def getNeurons(self):
        return self.neurons

    def forward(self):
        if hasattr(self.neurons, '__len__'):
            for n in self.neurons:
                n.forward()

class neuron:
    def __init__(self, bias):
        self.outputs = []
        self.value = 0
        self.bias = bias

    def addOutput(self, output):
        self.outputs.append(output)

    def getOutputs(self):
        return self.outputs

    def setValue(self, v):
        self.value = v

    def addValue(self, v):
        self.value += v

    def getValue(self):
        return self.value

    def forward(self):
        if hasattr(self.outputs, "__len__"):
            for o in self.outputs:
                o.setValue(self.value + self.bias)
                o.forward()

    def mutate(self):
        m = 0
        amount = 0
        threshold = 0.1
        while m < 100:
            chance = random.random()
            if chance < threshold:
                amount += chance
                threshold /= 10
                m -= 100
                amount *= -1
            m += 1

        self.addValue(amount)

class synapse:
    def __init__(self, weight):
        self.output = None
        self.value = 0
        self.weight = weight

    def addOutput(self, output):
        self.output = output

    def setValue(self, v):
        self.value = v

    def addValue(self, v):
        self.value += v

    def forward(self):
        if self.output != None:
            self.output.setValue(self.value + self.weight)

    def mutate(self):
        m = 0
        amount = 0
        threshold = 0.1
        while m < 100:
            chance = random.random()
            if chance < threshold:
                amount += chance
                threshold /= 10
                m -= 100
                amount *= -1
            m += 1

        self.addValue(amount)

class organism:
    def __init__(self, parents = None):
        self.alive = True
        self.id = None
        self.pX, self.pY = 0, 0
        self.pO = random.randrange(1, 359)
        self.mass = 0
        self.color1 = []
        self.color2 = []
        self.energy = 0
        self.age = 0
        self.sick = 0
        self.hunger = 0
        self.thirst = 0
        self.health = 0
        self.parents = None
        self.children = None
        self.closestResource = [None, 0, 0]
        self.closestOrganism = [None, 0, 0]
        
        # find attributes of nearest resource:
        # distance
        # direction
        # nearest organism:
        # distance
        # direction
        self.senses = [0, 0, 0, 0]

        inputLayer = len(self.senses)
        hiddenLayer = [3, 5]
        outputLayer = 3
        self.nn = neural_network(self, inputLayer, hiddenLayer, outputLayer)

        if parents == None:
            self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))

            self.pX = random.randrange(0, FIELDX)
            self.pY = random.randrange(0, FIELDY)

            self.color1 = [random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)]
            self.color2 = [random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)]
        else:
            if hasattr(parents, "__len__"):
                self.parents = parents

                self.id = parents[0].id[-5:] + parents[1].id[-5:]

                self.pX = parents[1].pX
                self.pY = parents[1].pY

                self.color1 = [((parents[0].color1[0] + parents[1].color1[0]) / 2), ((parents[0].color1[1] + parents[1].color1[1]) / 2), ((parents[0].color1[2] + parents[1].color1[2]) / 2)]
                self.color2 = [((parents[0].color2[0] + parents[1].color2[0]) / 2), ((parents[0].color2[1] + parents[1].color2[1]) / 2), ((parents[0].color2[2] + parents[1].color2[2]) / 2)]
            else:
                pass

        self.health = random.randrange(50, 100)
        self.energy = random.randrange(49, self.health)
        self.hunger = random.randrange(10, 30)
        self.thirst = random.randrange(10, 30)
        self.mass = 3

    def iterate(self):
        if self.alive:
            self.think()
            self.draw()

    def think(self):
        for e in entities:
            if e != self:
                if isinstance(e, resource):
                    if self.closestResource[0] == None:
                        self.closestResource[0] = e
                        self.closestResource[1] = distance(self.getXY(), self.closestResource[0].getXY())
                        self.closestResource[2] = getAngleBetweenPoints(self.getXY(), self.closestResource[0].getXY())
                    else:
                        if self.closestResource[0] != e:
                            if distance(self.getXY(), e.getXY()) < distance(self.getXY(), self.closestResource[0].getXY()):
                                self.closestResource[0] = e
                if isinstance(e, organism):
                    if self.closestOrganism[0] == None:
                        self.closestOrganism[0] = e
                        self.closestOrganism[1] = distance(self.getXY(), self.closestOrganism[0].getXY())
                        self.closestOrganism[2] = getAngleBetweenPoints(self.getXY(), self.closestOrganism[0].getXY())
                    else:
                        if self.closestOrganism[0] != e:
                            if distance(self.getXY(), e.getXY()) < distance(self.getXY(), self.closestOrganism[0].getXY()):
                                self.closestOrganism[0] = e
                        
                        

        self.senses = [self.closestResource[1], self.closestResource[2], self.closestOrganism[1], self.closestOrganism[2]]

        self.nn.forward(self.senses)

    def draw(self):
        pygame.draw.circle(FIELD, (self.color1[0], self.color1[1], self.color1[2]), (self.pX, self.pY), self.mass)
        pygame.draw.circle(FIELD, (self.color2[0], self.color2[1], self.color2[2]), (self.pX, self.pY), self.mass, 1)
        if self == selectedEntity:
            pygame.draw.circle(FIELD, (self.color1[0], self.color1[1], self.color1[2], (self.pX, self.pY), (self.mass + 3)))
            pygame.draw.line(FIELD, (self.color1[0], self.color1[1], self.color1[2]), (self.pX, self.pY), calc_point(self.pX, self.pY, self.pO, self.pO, 3), 3)

    def getXY(self):
        return self.pX, self.pY

    def move(self, s):
        self.pX, self.pY = calc_point(self.pX, self.pY, self.pO, s, 1)
        pass

    def rotate(self, a):
        self.pO += a
        if self.pO > 359:
            self.pO -= 359
        if self.pO < 0:
            self.pO += 359

    def mutate(self):
        for l in self.nn.getLayers():
            for n in l.getNeurons():
                n.mutate()
                for s in n.getOutputs():
                    s.mutate()



class resource:
    def __init__(self, pX = -1, pY = -1, mass = -1):
        if pX > -1:
            self.pX = pX
        else:
            self.pX = random.randrange(0, FIELDX)
        if pY > -1:
            self.pY = pY
        else:
            self.pY = random.randrange(0, FIELDY)

        if mass > -1:
            self.mass = mass * 0.9
        else:
            self.mass = 3

        self.color1 = [random.randrange(0, random.randrange(1, 100)), random.randrange(0, 255), random.randrange(0, random.randrange(1, 100))]

    def iterate(self):
        self.think()
        self.draw()

    def think(self):
        pass

    def draw(self):
        pygame.draw.circle(FIELD, (self.color1[0], self.color1[1], self.color1[2]), (self.pX, self.pY), self.mass)

    def getXY(self):
        return self.pX, self.pY

#----------MAIN

#-----PYGAME

FIELDX, FIELDY = 500, 500

pygame.init()
FONT = pygame.font.SysFont("monospace", 15)
WIN = pygame.display.set_mode((FIELDX + 215, FIELDY + 10))
pygame.display.set_caption("NNAI4")

#-----SIM FIELD
FIELD = pygame.Surface((FIELDY, FIELDX))
pygame.transform.rotate(FIELD, 90)

#-----STAT PANEL
PANEL = pygame.Surface((200, 500))

entities = []
selectedEntity = None

def updateSimulation():

    WIN.fill((220, 220, 220))
    WIN.blit(FIELD, (5, 5))
    FIELD.fill((225, 225, 225))
    
    for e in entities:
        e.iterate()

    if selectedEntity != None:
        pass

    pygame.display.update()

def main():

    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            #set exit handler
            if event.type == pygame.QUIT:
                run = False

            #handler for mouse clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                match event.button:
                    case 1: #left click
                        pass
                    case 2: #middle click
                        pass
                    case 3: #right click
                        pass
                    case 4: #scroll up
                        pass
                    case 5: #scroll down
                        pass

            #handler for key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    newOrganism = organism()
                    entities.append(newOrganism)
                    selectedEntity = newOrganism
                if event.key == pygame.K_r:
                    newResource = resource()
                    entities.append(newResource)
                    # selectedEntity = newResource
                if event.key == pygame.K_q:
                    if selectedEntity != None:
                        selectedEntity.rotate(-1)
                if event.key == pygame.K_w:
                    if selectedEntity != None:
                        selectedEntity.move(1)
                if event.key == pygame.K_e:
                    if selectedEntity != None:
                        selectedEntity.rotate(1)
                if event.key == pygame.K_m:
                    if selectedEntity != None:
                        selectedEntity.mutate()
                if event.key == pygame.K_c:
                    entities.clear()
                # pgkn = str(pygame.key.name(event.key))
                # pgkkc = str(pygame.key.key_code(pgkn))
                # print(pgkkc + " : " + pgkn)

            #handler for key holds
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed != None:
                pass

        updateSimulation()

    pygame.quit()

#----------RUN

if __name__ == "__main__":
    main()