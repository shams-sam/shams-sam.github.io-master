https://en.wikipedia.org/wiki/Discourse_analysis
https://en.wikipedia.org/wiki/Semiotics


### Discourse Analysis

Discourse Analysis or discourse studies is a general term used for a number of approaches used to analyze written, vocal, or sign language use, or any semiotic event.

> Semiotics (also called semiotic studies) is the study of meaning-making, the study of sign process (semiosis) and meaningful communication.

Objects of discourse analysis can be defined in terms of **coherent sequences** of sentences, propositions, speech, or turns-at-talk. It studies language beyond the boundary of sentences and prefers to analyze naturally occuring language.

> Discourse analysis differs from text linguistics in that former aims at revealing socio-psychological characteristics of the person rather than the text structure.

Discourse Analysis includes:

* **Dimensions of discourse** such as sounds, gestures, syntax, lexicon, style, rhetoric, meanings, speech acts, moves, strategies, turns, and other aspects of interaction.
* **Genres of discourse**: various types of discourse in politics, media, education, science etc.
* Relation between **discourse and syntactic structure**.
* Relation between **text and context**.
* Relation between **discourse and power**.
* Relation between **discourse and interaction**.
* Relation between **discourse and cognition and memory**.


### OCR

Steps in OCR:

- Image aquisition:
	- Quantization: generalized case of binarization
	- Compression

- Pre-processing:
	- Binarization: 
		- Global Threshold
		- Local Threshold
	- Filtering
		- Averaging
		- Min
		- Max
	- Morphological Operations
		- Erosion
		- Dilation
		- Opening
		- Closing
	- Skew Detection
		- Projection profiles
		- Hough Transform
		- Nearest neighborhood method
	- Thinning
	- Text line detection
		- Based on projections
		- Clustering of pixels

- Character Segmentation
	- Explicit
	- Implicit as a part of Classification
	- Other phases can provide contexual information

- Feature Extraction
	- Geometrical features
		- Loops
		- Strokes
	- Statistical features
		- Moments
	- PCA for dimensionality reduction

- Classification
	- Structural approach
	- Statistical approach
		- Bayesian classifier
		- Decision tree
		- Neural Network
		- Nearest neighborhood

- Post-processing
	- Ensemble multiple classifiers
		- Hierarchy
		- Cascade
		- Parallel
	- Lexical Processing
		- Markov models
		- Dictionary

	