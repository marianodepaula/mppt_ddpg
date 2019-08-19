import numpy as np
import matplotlib.pyplot as plt
import pickle


#Levantar un objeto guardado:
'''
f = open('obj.save', 'rb')
loaded_obj = cPickle.load(f)
print('altura del gato restored', loaded_obj.altura)
f.close()
'''

cant_objetos = np.load('cant_pruebas.npy') #ojo aca de no levantar pruebas viejas dado que esto se va guardando a mano, en funcion del numero de test...sino ponerlo a manopla

tension = []
potencia =[]
delta_V = []
corriente = []
temperatura = []
irradiancia = []
acciones = []
i=0
while i<=cant_objetos:
	#print('entramos al while')
	#Name = 'obj'+ str(i) +'.save'	
	#print('i =', i, 'Name=',Name)
	
	#Levantar un objeto guardado:
	#f = open(Name, 'rb')
	#f = open('obj1.save', 'rb')
	#loaded_data = cPickle.load(f)
	#f.close()

	Name = 'obj'+ str(i)+ '.save'
	print('Name es:', Name)
	with open(Name, 'rb') as data_file:
		loaded_data = pickle.load(data_file)

	if i == 0:
		first_index=0
	else:
		first_index =1

	tension = loaded_data.V[first_index:] + tension
	potencia = loaded_data.P[first_index:] + potencia
	delta_V = loaded_data.deltaV[first_index:] + delta_V
	corriente = loaded_data.I[first_index:] + corriente
	temperatura = loaded_data.Temp[first_index:] + temperatura
	irradiancia = loaded_data.Irr[first_index:] + irradiancia
	acciones = loaded_data.acciones[first_index:] + acciones

	i+=1

	#self.V.append(v), self.P.append(p), self.deltaV.append(dv), self.I.append(i), self.Temp.append(T), self.Irr.append(irr), self.acciones.append(accion)

plt.plot(tension,potencia)
plt.xlabel('V (v)')
plt.ylabel('P (w)')
plt.title('V-P curve')
plt.savefig('VPcurve' + '.png')
plt.show()
		
plt.plot(tension,corriente)
plt.xlabel('V (v)')
plt.ylabel('I (A)')
plt.title('V-I curve')
plt.savefig('VIcurve' + '.png')
plt.show()


plt.plot(tension)
plt.xlabel('t')
plt.ylabel('V (v)')
plt.savefig('Tesion' + '.png')
plt.show()

plt.plot(corriente)
plt.xlabel('t')
plt.ylabel('I (a)')
plt.savefig('Corriente' + '.png')
plt.show()

plt.plot(potencia)
plt.xlabel('t')
plt.ylabel('P (w)')
plt.savefig('Potencia' + '.png')
plt.show()

plt.plot(acciones)
plt.xlabel('t')
plt.ylabel('acciones (\deltaV)')
plt.title('actions')
plt.savefig('Acciones' + '.png')
plt.show()

plt.plot(temperatura)
plt.xlabel('t')
plt.ylabel('(ºC)')
plt.title('Temperature profile')
plt.savefig('Temperatura' + '.png')
plt.show()

plt.plot(irradiancia)
plt.xlabel('t')
plt.ylabel('(Irradiance)')
plt.title('Solar irradiance profile')
plt.savefig('Irradiancia' + '.png')
plt.show()

