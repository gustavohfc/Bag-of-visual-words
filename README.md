Alunos: Fábio Costa Farias Marques, Gustavo Henrique Fernandes Carvalho

Matriculas: 140039082, 140021671


# Algoritmo
A versão de Python utilizada foi a 3.5.2.

# Pacotes Python requisitados
	-> 	opencv-python
	-> 	opencv-python-contrib
	-> 	numpy
	->	scipy
	->	sklearn

Para executar o programa para tanto treinar, quanto testar com as imagens fornecidas
```console
$ python bag_of_visual_words.py
```

O programa tem a opção de carregar o codebook e os histogramas de classe previamente calculados ao invés de realizar o treinamento completo, desde que os arquivos "codeBook.pkl" e "histograms.pkl" estejam na mesma pasta do programa.

Para o correto funcionamento do programa, uma pasta chamada "Images" deve estar no mesmo diretório do programa. Dentro dela estarão todas a base necessária para a execução, sendo que a hierarquia deve ser:

- Images;
	- Test;
		- Dentro desta pasta deverá haver uma pasta para cada classe a ser identificada, tal pasta contendo as imagens de teste.
	- Train;
		- Dentro desta pasta deverá haver uma pasta para cada classe a ser identificada, tal pasta contendo as imagens para treino e desenvolvimento dos histogramas de classe.