# ori

Na osnovu numeričkih podataka je potrebno odrediti da li postoji aritmija i ako postoji klasifikovati je u 16 postojećih vrsta, od kojih 16-ta predstavlja neklasifikovane aritmije.

Algoritmi:

Neuronska mreža za prepoznavanje postojanja aritmije
Levenberg-Marquardt algoritma(metoda najmanjih kvadrata) za klasifikaciju aritmija
Dataset skup sa sajta https://archive.ics.uci.edu/ml/datasets/Arrhythmia. Potrebno je izbaciti redove sa nedostajućim vrednostima i redove čiji su sadržaj sve nule. Takođe je potrebno malo smanjiti broj kolona, koristiće se samo kolone(atributi) koji su najviše povezani sa pojavljivanjem aritmije.

Kao metrika za poređenje koristiće se procenat tačnog određivanja pojave aritmije i procenat tačne klasifikacije aritmije. Dataset će biti podeljen na tri dela, najveći deo će biti podaci za treniranje neuronske mreže(oko 75%), podaci za validaciju(oko 15%) i test podaci(oko 10%).

Pokretanje:
Kako bi se ponovo istrenirale neuronske mreže potrebno je obrisati .net fajlove, ako želite da učitate već postojeće mreže samo pokrenuti ArrhythmiaRecognition.py.
Program ispisuje koje se mreže koriste, sa kojim podacima i sa kojom tačnošću. Na kraju nudi da se unesu atributi(jedan red iz test.txt fajla iz [] zagrada) i ispisuje rezultate neuronske mreže.

