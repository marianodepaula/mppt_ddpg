import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle


#Levantar un objeto guardado:
'''
f = open('obj.save', 'rb')
loaded_obj = cPickle.load(f)
print('altura del gato restored', loaded_obj.altura)
f.close()
'''

cant_objetos = 0 # np.load('cant_pruebas.npy') #ojo aca de no levantar pruebas viejas dado que esto se va guardando a mano, en funcion del numero de test...sino ponerlo a manopla

tension = []
potencia =[]
delta_V = []
corriente = []
temperatura = []
irradiancia = []
acciones = []

i=0
Name = 'obj'+ str(i) +'.save'	
print('i =', i, 'Name=',Name)

#Levantar un objeto guardado:
f = open('obj0.save', 'rb')
loaded_data = cPickle.load(f)
f.close()

if i == 0:
	first_index = 0
else:
	first_index =1

tension = loaded_data.V[first_index:] + tension
potencia = loaded_data.P[first_index:] + potencia
delta_V = loaded_data.deltaV[first_index:] + delta_V
corriente = loaded_data.I[first_index:] + corriente
temperatura = loaded_data.Temp[first_index:] + temperatura
irradiancia = loaded_data.Irr[first_index:] + irradiancia
acciones = loaded_data.acciones[first_index:] + acciones

'''
i=1
Name = 'obj'+ str(i) +'.save'	
print('i =', i, 'Name=',Name)

#Levantar un objeto guardado:
g = open(Name, 'rb')
loaded_data = cPickle.load(g)
g.close()

if i == 0:
	first_index = 0
else:
	first_index =1

tension = loaded_data.V[first_index:] + tension
potencia = loaded_data.P[first_index:] + potencia
delta_V = loaded_data.deltaV[first_index:] + delta_V
corriente = loaded_data.I[first_index:] + corriente
temperatura = loaded_data.Temp[first_index:] + temperatura
irradiancia = loaded_data.Irr[first_index:] + irradiancia
acciones = loaded_data.acciones[first_index:] + acciones

#self.V.append(v), self.P.append(p), self.deltaV.append(dv), self.I.append(i), self.Temp.append(T), self.Irr.append(irr), self.acciones.append(accion)
'''

plt.plot(tension,potencia)
plt.xlabel('V (v)')
plt.ylabel('P (w)')
plt.title('V-P curve')
plt.savefig('VPcurve' + '.png')
plt.show()

