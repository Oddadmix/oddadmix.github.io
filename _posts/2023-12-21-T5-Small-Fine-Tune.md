## How to fine-tune t5-small to generate color hex

As this was my first llm to be fine-tuned, I wanted to take on a task that is relatively simple.

To start, a plan must be drafted to kind of stir how this model will be fine-tuned, the result here was to generate a 4 colors palette using user input, the output will be 4 colours hex.

## The Plan
- Find a model that already does that
- Curate/Find a dataset 
- **Prepare the data**
- Train the model
- Evaluate and retierate until converage ***(Size is important not always)***

## The Search
As the plan was clear and with the first step, I started my journey on hugging face to find a model that already does that, give it some text and it returns back with hex color or any sort of colour representation.
After a couple of hours it seem that it wasn't an easy ask, I wasn't able to find a model that specialize in that sort of task ***(Is it color or colour)***

## The Data
The second step was to find data if a model doesn't exist already to start the fine-tuning process. Again not an easy task, as I was looking for text to 4 hex colors, I wasn't able to find such data on hugging face or in my search in general. So I rolled my sleves and headed to chatgpt, and asked `suggest dataset to finetune an llm to provide 4 colors when given a palette name` and here was the output
```
| Palette Name    | Color 1    | Color 2    | Color 3    | Color 4    |
|-----------------|------------|------------|------------|------------|
| Spring Meadow   | #8FD400    | #62C100    | #3AA300    | #207C00    |
| Ocean Blues     | #007BFF    | #0056B3    | #003D66    | #001F33    |
| Sunset Shades   | #FF4500    | #FF7E00    | #FFAC00    | #FFD700    |
| Autumn Hues     | #FFA500    | #D2691E    | #8B4513    | #8B0000    |
| Pastel Dream    | #FFD700    | #FFA07A    | #FFB6C1    | #87CEEB    |
```
Looks promising, not quite, asking chat gpt to generate a couple of hundered just ended up having lots of duplicates. There has to be another solution.

## The Scrapping
I love scrapping, it's one of the tools that every developer should have, through scrapping you get to be exposed to different kinds of tech, through the target websites. I explored a couple of color palette websites and started to look for ones that provide both a color palette and theme name or a description of that colors. 

I finally find one of the websites that does that and compiled around ~800 rows.
```python
import requests
from bs4 import BeautifulSoup
import json

color_palettes = []

for i in  range(1, 800):
	URL =  "https://thewebsite"  +  str(i) +  "/"
	page = requests.get(URL)
	soup = BeautifulSoup(page.content, "html.parser")
	results = soup.select("#colors")
	## loop through all the colors and store them in an array
	colors = []
	for result in results:
		colors.append(result['data-clipboard-text'])
	## if no colors, skip
	if  len(colors) ==  0:
		continue
	description = soup.select_one(".text_content p").text
	tags = soup.select_one(".tags").text
	## add to object
	color_palettes.append({
	"colors": colors,
	"description": description,
	"tags": tags	
	})
	##Saving every iteration to avoid issues and losing all data
	with  open('color_palettes.json', 'w') as outfile:
		json.dump(color_palettes, outfile)
```
