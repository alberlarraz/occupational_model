# -*- coding: utf-8 -*-
import numpy as np
import matplotlib, math
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import PIL
from PIL import Image
from PIL import ImageDraw
import sys

def main(inputs):

### Inputs:
#
	d = float(inputs[0])	# Particle diameter [nm]
	den = float(inputs[1])		# Density [g/cm3]
	wt = float(inputs[2])*0.01  # fraction of pure NM (weight) [%] -> [x/1]
	m = float(inputs[3])*1.e6	# Mass used [g] -> [ug]
	td = float(inputs[4])	# Task duration [min]
	tg = float(inputs[5])	# Generation rate [min]
	nrep = int(inputs[6])	# Number of repetitions [#]
	V = float(inputs[7])	# Room volume [m3]
	ach = float(inputs[8])	# Air changes [#/h]
	rmm_eng = int(inputs[9])	# Risk management measure - Engineering control [position in list]
	rmm_rpe = int(inputs[10])	# Risk management measure - Respiratory protective equipment [position in list]
	rmm_ocu = int(inputs[11])	# Risk management measure - Ocular protection [position in list]
	rmm_hand = int(inputs[12])	# Risk management measure - Dermal protection (hands) [position in list]
	rmm_body = int(inputs[13])	# Risk management measure - Dermal protection (body) [position in list]
	release_id = int(inputs[14])	# Activity release rate [position in list]
	release = float(inputs[15])	# Activity release rate [0-1]
	QL=float(inputs[16]) # Local exhaust ventilation rate [m^3/min]
	QR=float(inputs[17]) # Room recirculation system ventilation rate [m^3/min]
	epsilonRF=float(inputs[18])*0.01 # General ventilation recirculation filtration efficiency [%]
	epsilonL=float(inputs[19])*0.01 # Fraction of the source emissions inmediately captured by the local exhaust [%]
	epsilonLF=float(inputs[20])*0.01 # Local exhaust filtration efficiency [%]
	
	### Constants:
	pi=np.pi	# Pi
	betta = 5.	# NF ventilation rate (fixed): [m3/min]
	step = 0.1	# Time step for evaluation of the model [min]
	cbg = 0.	# background concentration [mg/m^3]
	Vn = 8.		# Fixed to 8 m^3
### Generation rate (emission rate). Changes depending on NM and process characteristics
#
	gamma = (nrep*tg)/(nrep*td)	# fraction of time action is taking place
	if release_id == 162:
		pond = release
	else:
		release_data = pd.read_csv('release_rates.csv', sep=";", header=0, index_col='ID')
		pond = float(release_data.loc[release_id, 'Ponderation']) # Multiplier to obtain nm release amount (mg)
	G = (m*wt*pond)/td	# mass flow (emission Rate) [mg/min]
	Vf = V-Vn	# Volume of the FF as total (volume - volume of NF)
	Q = (ach/60)*V # Room ventilation rate [m^3/min] 

### Risk management measures:
#
	rmms_data = pd.read_csv('rmms.csv', sep=";", header=0, index_col='ID')
	# Engineering control
	rmm_eng_pr = float(rmms_data.loc[rmm_eng, 'Ponderation'])
	# Respiratory Protective Equipment
	rmm_rpe_pr = float(rmms_data.loc[rmm_rpe, 'Ponderation'])
	# Ocular protection
	rmm_ocu_pr = float(rmms_data.loc[rmm_ocu, 'Ponderation'])
	# Dermal protection (hands)
	rmm_hand_pr = float(rmms_data.loc[rmm_hand, 'Ponderation'])
	# Dermal protection (body)
	rmm_body_pr = float(rmms_data.loc[rmm_body, 'Ponderation'])

### Model definition
#
	Model=201;

	# Model 200: Natural ventilation, steady state
	# Model 201: Natural ventilation, transient
	# Model 202: Natural ventilation and air recirculation, steady state
	# Model 203: Natural ventilation and air recirculation, transient
	# Model 204: Natural ventilation and near field air exhaust, steady state
	# Model 205: Natural ventilation and near field air exhaust, transient
	# Model 206: Natural ventilation, air recirculation and near field air exhaust, steady state
	# Model 207: Natural ventilation, air recirculation and near field air exhaust, transient
	# Model 208: Natural ventilation and near field air exhaust to far field, steady state
	# Model 209: Natural ventilation and near field air exhaust to far field, transient
	# Model 210: Natural ventilation, air recirculation and near field air exhaust to far field, steady state
	# Model 200: Natural ventilation, air recirculation and near field air exhaust to far field, transient

### Model parameters
#

	if(Model==205 or Model==207 or Model==209 or Model==211):
		bettai=QL+betta
		a = Vf*Vn
		def b(x):
			 return (bettai+x)*Vn+bettai*Vf
		def c(x):
			if (Model==205 or Model==207):
				return bettai*(bettai+x)-betta*bettai
			else:
				return bettai*(bettai+x)-bettai*(betta+QL*(1-epsilonLF))
		if(Model==205 or Model==209):
			r1=(-b(Q)+math.sqrt(b(Q)**2-(4*a*c(Q))))/(2*a)
			r2=(-b(Q)-math.sqrt(b(Q)**2-(4*a*c(Q))))/(2*a)
		else:
			r1=(-b(Q+epsilonRF*QR)+math.sqrt(b(Q+epsilonRF*QR)**2-(4*a*c(Q+epsilonRF*QR))))/(2*a)
			r2=(-b(Q+epsilonRF*QR)-math.sqrt(b(Q+epsilonRF*QR)**2-(4*a*c(Q+epsilonRF*QR))))/(2*a)
			
	def P1(x,y):
		return (x*Vn+y)/y

	def P2(x,y):
		 return (betta*x+y*Vn*(betta+x))/(betta*x*Vn*(lambda1-lambda2))

	def P3(x,y):
		return x*Vn+y

	def P4(Cf_zero,Cn_zero):
		return (betta*(Cf_zero-Cn_zero)-(lambda2*Vn*Cn_zero))/(betta*Vn*(lambda1-lambda2))

	def P5(Cf_zero,Cn_zero):
		return (betta*(Cn_zero-Cf_zero)-(lambda1*Vn*Cn_zero))/(betta*Vn*(lambda1-lambda2))

	def P6(x,y,z):
		return (bettai*x-y*(bettai+Vn*z))/(Vn*(r2-r1))

	def P7(x):
		return x

### Model equations 
#
	if(Model==200):
		Cf_ss=(gamma*G)/Q
		Cn_ss=Cf_ss+(gamma*G)/betta

	elif(Model==201):
		Cf_ss=(G)/Q
		Cn_ss=Cf_ss+G/betta
		k1 = (betta*Vf+Vn*(betta+Q))/(Vn*Vf)
		k2 = (betta*Q)/(Vn*Vf) 
		lambda1=0.5*((-k1)+((k1**2)-4*k2)**(0.5))
		lambda2=0.5*((-k1)-((k1**2)-4*k2)**(0.5))
		def Cf_g(t):
			return Cf_ss+G*P1(lambda1,betta)*P2(Q,lambda2)*np.exp(lambda1*t)-G*P1(lambda2,betta)*P2(Q,lambda1)*np.exp(lambda2*t)
		def Cn_g(t):
			return Cn_ss+G*P2(Q,lambda2)*np.exp(lambda1*t)-G*P2(Q,lambda1)*np.exp(lambda2*t)
		def Cf_d(t,Cf_zero, Cn_zero):
			return P3(lambda1,betta)*P4(Cf_zero,Cn_zero)*np.exp(lambda1*(t))+P3(lambda2,betta)*P5(Cf_zero,Cn_zero)*np.exp(lambda2*(t))
		def Cn_d(t,Cf_zero, Cn_zero):
			return betta*P4(Cf_zero,Cn_zero)*np.exp(lambda1*(t))+betta*P5(Cf_zero,Cn_zero)*np.exp(lambda2*(t))

	elif(Model==202):
		Cf_ss=(gamma*G)/(Q+QL*epsilonRF)
		Cn_ss=Cf_ss+(gamma*G)/Q
		Crf=Cf_ss*(1-epsilonRF)

	elif(Model==203):
		Cf_ss=(G)/(Q+QR*epsilonRF)
		Cn_ss=Cf_ss+(G)/betta
		k1 = (betta*Vf+Vn*(betta+(Q+QR*epsilonRF)))/(Vn*Vf)
		k2 = (betta*(Q+QR*epsilonRF))/(Vn*Vf) 
		lambda1=0.5*((-k1+(k1**2-4*k2)**(0.5)))
		lambda2=0.5*((-k1-(k1**2-4*k2)**(0.5)))
		def Cf_g(t):
			return Cf_ss+G*P1(lambda1,betta)*P2(Q+epsilonRF*QR,lambda2)*np.exp(lambda1*t)-G*P1(lambda2,betta)*P2(Q+epsilonRF*QR,lambda1)*np.exp(lambda2*t)
		def Cn_g(t):
			return Cn_ss+G*P2(Q+epsilonRF*QR,lambda2)*np.exp(lambda1*t)-G*P2(Q+epsilonRF*QR,lambda1)*np.exp(lambda2*t)
		def Cf_d(t,Cf_zero, Cn_zero):
			return P3(lambda1,betta)*P4(Cf_zero,Cn_zero)*np.exp(lambda1*(t))+P3(lambda2,betta)*P5(Cf_zero,Cn_zero)*np.exp(lambda2*(t))
		def Cn_d(t,Cf_zero, Cn_zero):
			return betta*P4(Cf_zero,Cn_zero)*np.exp(lambda1*(t))+betta*P5(Cf_zero,Cn_zero)*np.exp(lambda1*(t))

	elif(Model==204):
		epsilonN=QL/(QL+betta)
		Cf_ss=(gamma*G*(1-epsilonL)*(1-epsilonN))/(Q+QL)
		Cn_ss=Cf_ss+(gamma*G*(1-epsilonL)*epsilonN)/QL
		Cle=Cn_ss+(gamma*G*epsilonL)/QL

	elif(Model==205):
		epsilonN=QL/(QL+betta)
		Cf_ss=(G*(1-epsilonL)*(1-epsilonN))/(Q+QL)
		Cn_ss=Cf_ss+(G*(1-epsilonL)*epsilonN)/QL
		Cle=Cn_ss+(G*epsilonL)/QL
		def Cf_g(t):
			return Cf_ss+P6(Cf_ss,Cn_ss,r2)*P1(r1,bettai)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*P1(r2,bettai)*np.exp(r2*t)
		def Cn_g(t):
			return Cn_ss+P6(Cf_ss,Cn_ss,r2)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*np.exp(r2*t)
		def Cf_d(t,Cf_zero, Cn_zero):
			return P1(r1,bettai)*-P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t))+P1(r2,bettai)*P6(Cf_zero,Cn_zero,r1)*np.exp(r2*(t))
		def Cn_d(t,Cf_zero, Cn_zero):
			return -P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t))+P6(Cn_zero,Cn_zero,r1)*np.exp(r2*(t))

	elif(Model==206):
		epsilonN=QL/(QL+betta)
		Cf_ss=(gamma*G*(1-epsilonL)*(1-epsilonN))/(Q+epsilonRF*QR+QL)
		Cn_ss=Cf_ss+(gamma*G*(1-epsilonL)*epsilonN)/QL
		Cle=Cf_ss+(gamma*G*epsilonL)/QL
		Crf=Cf_ss*(1-epsilonRF)

	elif(Model==207):
		epsilonN=QL/(QL+betta)
		Cf_ss=(G*(1-epsilonL)*(1-epsilonN))/(Q+epsilonRF+QL)
		Cn_ss=Cf_ss+(G*(1-epsilonL)*epsilonN)/QL
		Cle=Cn_ss+(G*epsilonL)/QL
		Crf=Cf_ss*(1-epsilonRF)
		def Cf_g(t):
			return Cf_ss+P6(Cf_ss,Cn_ss,r2)*P1(r1,bettai)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*P1(r2,bettai)*np.exp(r2*t)
		def Cn_g(t):
			return Cn_ss+P6(Cf_ss,Cn_ss,r2)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*np.exp(r2*t)
		def Cf_d(t,Cf_zero, Cn_zero):
			return P1(r1,bettai)*-P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t))+P1(r2,bettai)*P6(Cf_zero,Cn_zero,r1)*np.exp(r2*(t))
		def Cn_d(t,Cf_zero, Cn_zero):
			return -P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t))+P6(Cf_zero,Cn_zero,r1)*np.exp(r2*(t))

	elif(Model==208):
		epsilonN=QL/(QL+betta)
		Cf_ss=(gamma*G*(1-epsilonL*epsilonLF-epsilonN*epsilonLF*(1-epsilonL)))/(Q+epsilonLF*QL)
		Cn_ss=Cf_ss+(gamma*G*(1-epsilonL)*epsilonN)/QL
		Cle=Cn_ss+(gamma*G*epsilonL)/QL
		Clf=Cle*(1-epsilonRF)

	elif(Model==209):
		epsilonN=QL/(QL+betta)
		Cf_ss=(G*(1-epsilonL*epsilonLF-epsilonN*epsilonLF*(1-epsilonL)))/(Q+epsilonLF*QL)
		Cn_ss=Cf_ss+(G*(1-epsilonL)*epsilonN)/QL
		Cle=Cn_ss+(G*epsilonL)/QL
		Crf=Cf_ss*(1-epsilonRF)
		def Cf_g(t):
			return Cf_ss+P6(Cf_ss,Cn_ss,r2)*P1(r1,bettai)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*P1(r2,bettai)*np.exp(r2*t)
		def Cn_g(t):
			return Cn_ss+P6(Cf_ss,Cn_ss,r2)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*np.exp(r2*t)
		def Cf_d(t,Cf_zero, Cn_zero):
			return P1(r1,bettai)*-P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t-tg))+P1(r2,bettai)*P6(Cf_zero,Cn_zero,r1)*np.exp(r2*(t-tg))
		def Cn_d(t,Cf_zero, Cn_zero):
			return -P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t))+P6(Cn_zero,Cn_zero,r1)*np.exp(r2*(t))

	elif(Model==210):
		epsilonN=QL/(QL+betta)
		Cf_ss=(gamma*G*(1-epsilonL*epsilonLF-epsilonN*epsilonLF*(1-epsilonL)))/(Q+epsilonRF*QR+epsilonLF*QL)
		Cn_ss=Cf_ss+(gamma*G*(1-epsilonL)*epsilonN)/QL
		Cle=Cn_ss+(gamma*G*epsilonL)/QL
		Crf=Cf_ss*(1-epsilonRF)

	elif(Model==211):
		epsilonN=QL/(QL+betta)
		Cf_ss=(G*(1-epsilonL*epsilonLF-epsilonN*epsilonLF*(1-epsilonL)))/(Q+epsilonLF*QL)
		Cn_ss=Cf_ss+(G*(1-epsilonL)*epsilonN)/QL
		Cle=Cn_ss+(G*epsilonL)/QL
		Crf=Cf_ss*(1-epsilonRF)
		def Cf_g(t):
			return Cf_ss+P6(Cf_ss,Cn_ss,r2)*P1(r1,bettai)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*P1(r2,bettai)*np.exp(r2*t)
		def Cn_g(t):
			return Cn_ss+P6(Cf_ss,Cn_ss,r2)*np.exp(r1*t)-P6(Cf_ss,Cn_ss,r1)*np.exp(r2*t)
		def Cf_d(t,Cf_zero, Cn_zero):
			return P1(r1,bettai)*-P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t-tg))+P1(r2,bettai)*P6(Cf_zero,Cn_zero,r1)*np.exp(r2*(t-tg))
		def Cn_d(t,Cf_zero, Cn_zero):
			return -P6(Cf_zero,Cn_zero,r2)*np.exp(r1*(t))+P6(Cf_zero,Cn_zero,r1)*np.exp(r2*(t))

	else:
		print('The model does not exist')

	if (Model==201 or Model==203 or Model==205 or Model==207 or Model==209 or Model==211):
		CN2box = CN2box_f = []
		CF2box = CF2box_f = []
		n=0
		t=0
		while (n < nrep):
			while (t < td):
				if (t <= tg):               # generation phase
					CF2box.append(Cf_g(t))
					CN2box.append(Cn_g(t))
					if (tg-t < step):     # when t=tg is the first step of the decay phase.
						Cf_zero = Cf_g(t)
						Cn_zero = Cn_g(t)
				else:                       # decay phase
					CF2box.append(Cf_d(t,Cf_zero, Cn_zero))
					CN2box.append(Cn_d(t,Cf_zero, Cn_zero))
		
				t=round(t+step, 1)
		
			if (n==0):
				CN2box_f = CN2box
				CF2box_f = CF2box
			else:
				CN2box_f = CN2box_f+CN2box
				CF2box_f = CF2box_f+CF2box
			n=n+1

	### Concentration in ug/m3
	#
		C2boxNAv = integrate.simps(CN2box_f, dx=step, even='avg')/(nrep*td)
		C2boxFAv = integrate.simps(CF2box_f, dx=step, even='avg')/(nrep*td)
		
	### Concentration in particle number (for spherical particles)
	#
		Cp2boxN = [x/((pi*(d**3)*den)/(6*1e15))*1e-6 for x in CN2box_f]  # particles/cm3
		Cp2boxNAv = integrate.simps(Cp2boxN, dx=step, even='avg')/(nrep*td)
		Cp2boxF = [x/((pi*(d**3)*den)/(6*1e15))*1e-6 for x in CF2box_f]  # particles/cm3
		Cp2boxFAv = integrate.simps(Cp2boxF, dx=step, even='avg')/(nrep*td)

	### Application of risk management measures
	#
		print("Average concentration in Near Field: %.3e ug/m3\t %.3e #/cm3" %(C2boxNAv, Cp2boxNAv))
		print("Average concentration in Far Field: %.3e ug/m3\t %.3e #/cm3\n" %(C2boxFAv, Cp2boxFAv))
		
		# Application of LEV system
		Cp2boxNAv_LEV = Cp2boxNAv*rmm_eng_pr
		print("Average concentration in Near Field with selected LEV: %.3e #/cm3\n" %Cp2boxNAv_LEV)

		# Application of PPEs
		Cp2boxNAv_Resp = Cp2boxNAv_LEV*rmm_rpe_pr
		Cp2boxNAv_Ocu = Cp2boxNAv_LEV*rmm_ocu_pr
		Cp2boxNAv_Hand = Cp2boxNAv_LEV*rmm_hand_pr
		Cp2boxNAv_Body = Cp2boxNAv_LEV*rmm_body_pr
		print("Respiratory exposure with selected EPI: %.3e #/cm3" %Cp2boxNAv_Resp)
		print("Ocular exposure with selected EPI: %.3e #/cm3" %Cp2boxNAv_Ocu)
		print("Dermal exposure (hands) with selected EPI: %.3e #/cm3" %Cp2boxNAv_Hand)
		print("Dermal exposure (body) with selected EPI: %.3e #/cm3" %Cp2boxNAv_Body)

		comm = "Occupational Model"
		
		### Mass concentration
		plt.figure(1)
		x = np.arange(0,nrep*(td),step) # X-axis tics are spaced 'step' mins. until 'td'
		plt.title(str(comm))
		plt.plot (x, CN2box_f, linestyle='--', label="NF")
		plt.plot (x, CF2box_f, label="FF")
		plt.xlabel("$t\ [min]$")
		plt.ylabel("$C\ [ug/m^3]$")
		#plt.axhline(y=C2boxNAv, color='r', linestyle='-', label="Mean NF")
		#plt.axhline(y=C2boxFAv, color='g', linestyle='-', label="Mean FF")
		plt.ylim(ymin=0)  
		plt.legend(loc=1)
		plt.savefig("mass_concs.png")

		### Particle concentration
		plt.figure(2)
		x = np.arange(0,nrep*(td),step)  # X-axis tics are spaced 'step' mins. until 'td'
		plt.title(str(comm))
		plt.plot (x, Cp2boxN, linestyle='--', label="NF")
		plt.plot (x, Cp2boxF, label="FF")
		plt.xlabel("$t\ [min]$")
		plt.ylabel("$C\ [part/cm^3]$")
		#plt.axhline(y=Cp2boxNAv, color='r', linestyle='-', label="Mean NF")
		#plt.axhline(y=Cp2boxFAv, color='g', linestyle='-', label="Mean FF")
		plt.ylim(ymin=0)  
		plt.legend(loc=1)
		plt.savefig("partic_concs.png")
	else:
		print("This model is stacionary")
		print('Average concentration in Near Field: %.3e ug/m3' %Cn_ss)
		print('Average concentration in Far Field: %.3e ug/m3' %Cf_ss)

	

if __name__ == "__main__":
	main([x for x in sys.argv[1:22]])

