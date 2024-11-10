from transformers import pipeline, AdamW, AutoTokenizer as AT, AutoModelForSequenceClassification as AMSC
import torch
from datasets import load_dataset

def model(words):
	"""
	_______________________________________________________
	Parameters:
	words - 1D Array with 16 shuffled words
	strikes - Integer with number of strikes
	isOneAway - Boolean if your previous guess is one word away from the correct answer
	correctGroups - 2D Array with groups previously guessed correctly
	previousGuesses - 2D Array with previous guesses
	error - String with error message (0 if no error)

	Returns:
	guess - 1D Array with 4 words
	endTurn - Boolean if you want to end the puzzle
	_______________________________________________________
	"""

	#Your Code here
	pipe = pipeline("zero-shot-classification")
	possibilites = []
	for i in words:
		for j in words[1:]:
			for k in words[2:]:
				for l in words[3:]:
					sample = pipe(
						"Words that are similar to each other",
						candidate_labels=[i,j,k,l]
					)
					possibilites.append(sample)

	max = 0
	temp = ""
	for possible in possibilites:
		if possible["scores"][0] > max:
			max = possible["scores"][0]
			temp = possible
	# Good Luck!

	# Example code where guess is hard-coded
	guess = temp["labels"] # 1D Array with 4 elements containing guess
	print(guess)
	endTurn = False # True if you want to end puzzle and skip to the next one

	return guess, endTurn

model(["BENT","GNARLY","TWISTED","WARPED", "LICK","OUNCE","SHRED","TRACE", "EXPONENT","POWER","RADICAL","ROOT"])

def starter_code():
    return None