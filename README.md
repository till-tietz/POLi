# POLi 

POLi is an experimental natural language processing model designed to predict the ideological position expressed in german political text. It utilizes custom word embeddings fed into a CNN neural network architecture to generate ideology predictions. POLi was trained on ideology labeled german party manifesto text data from The Manifesto Project ¹. As the data set used to train POLi is comparitively small and domain specific, the model will likely not generalize well beyond the immediate context of german political texts (i.e. parliamentary bills, policy proposals, parliamentary inquiries etc.). Within this domain; however, initial tests indicate promising performance. 

## Current features & cotent

* data processing and model training code 
* shiny app to interact with the trained model

## Future work 

* improving model architecture 
* incorporating Austrian manifesto data 
* implementing sentiment prediction for each ideology/topic label 


## References 

¹ Volkens, Andrea / Burst, Tobias / Krause, Werner / Lehmann, Pola / Matthiess Theres / Merz, Nicolas / Regel, Sven / Wessels, Bernhard / Zehnter, Lisa (2020): The Manifesto Data Collection. Manifesto Project (MRG/CMP/MARPOR). Version 2020b. Berlin: Wissenschaftszentrum Berlin fuer Sozialforschung (WZB)
