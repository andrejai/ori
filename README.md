# ori

Na osnovu numeričkih podataka je potrebno odrediti da li postoji aritmija i ako postoji klasifikovati je u 16 postojećih vrsta, od kojih 16-ta predstavlja neklasifikovane aritmije.

Algoritmi:

Neuronska mreža za prepoznavanje postojanja aritmije, resilient backpropagation funkcija se koristi u neuronskoj mreži. Postoji jedan ulazni sloj, jedan skriveni i jedan izlazni sloj.
Korišćene su četiri neuronske mreže:
	-Neuronska mreža sa 30 neurona u skrivenom sloju, 50 epoha, trenirana sa svim atributima iz data set-a 
	-Neuronska mreža sa 32 neurona u skrivenom sloju, 100 epoha, trenirana sa 18 atributa dobijenih na osnovu istraživanja http://www.sciencedirect.com/science/article/pii/S2212017313004933
	-Neuronska mreža sa 6 neurona u skrivenom sloju, 500 epoha, trenirana sa 18 atributa, rezultat je samo da li aritmija postoji ili ne 
	-Neuronska mreža sa 30 neurona u skrivenom sloju, 200 epoha, trenirana sa 241 atributom dobijenim na osnovu korelacije(svi sa korelacijama manjim od 0,6)
	
Dataset skup sa sajta https://archive.ics.uci.edu/ml/datasets/Arrhythmia. Potrebno je izbaciti redove sa nedostajućim vrednostima i redove čiji su sadržaj sve nule. Takođe je potrebno malo smanjiti broj kolona, koristiće se samo kolone(atributi) koji su najviše povezani sa pojavljivanjem aritmije.

Kao metrika za poređenje koristiće se procenat tačnog određivanja pojave aritmije i procenat tačne klasifikacije aritmije. Dataset će je podeljen na dva dela, najveći deo će biti podaci za treniranje neuronske mreže(oko 80%) i test podaci(oko 20%).

Pokretanje:
Kako bi se ponovo istrenirale neuronske mreže potrebno je obrisati .net fajlove, ako želite da učitate već postojeće mreže samo pokrenuti ArrhythmiaRecognition.py.
Program ispisuje koje se mreže koriste, sa kojim podacima i sa kojom tačnošću. Na kraju nudi da se unesu atributi(jedan red iz test.txt fajla iz [] zagrada) i ispisuje rezultate neuronske mreže.

