from vllm import LLM, SamplingParams
import re

def extract_unique_vocab(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)  

prompts = [
    "Develop a series of three bedtime stories, each emphasizing key values like friendship, kindness, honesty, bravery, and patience.",
    "Start each story with a moral that encapsulates its main lessons at the top, followed by a narrative that becomes progressively more complex.",
    "The first story should contain at least 15 sentences and 200 unique words.", 
    "Increase the number of sentences and enhance the vocabulary richness in each subsequent story to match the advancing learning stages of children.",
    "This approach allows parents to tailor the complexity based on their child's understanding.",
    "Ensure accurate counting of unique vocabulary in each story to track the progression in language complexity."
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    story = output.outputs[0].text
    unique_vocab = extract_unique_vocab(story)
    print(f"Story {i+1}: {story!r}")
    print(f"Actual Unique Vocabulary Count: {len(unique_vocab)}\n")


# import re

# class StoryCurriculum:
#     def __init__(self, model):
#         self.llm = model
#         self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)

#     def extract_unique_vocab(self, text):
#         words = re.findall(r'\b\w+\b', text.lower())
#         return set(words)

#     def generate_stories(self):
#         prompts = [
#             "Develop a series of three bedtime stories, each emphasizing key values like friendship, kindness, honesty, bravery, and patience.",
#             "Start each story with a moral that encapsulates its main lessons at the top, followed by a narrative that becomes progressively more complex.",
#             "The first story should contain at least 15 sentences and 200 unique words.",
#             "Increase the number of sentences and enhance the vocabulary richness in each subsequent story to match the advancing learning stages of children.",
#             "This approach allows parents to tailor the complexity based on their child's understanding.",
#             "Ensure accurate counting of unique vocabulary in each story to track the progression in language complexity."
#         ]

#         outputs = self.llm.generate(prompts, self.sampling_params)

#         for i, output in enumerate(outputs):
#             story = output.outputs[0].text
#             unique_vocab = self.extract_unique_vocab(story)
#             print(f"Story {i+1}: {repr(story)}")
#             print(f"Actual Unique Vocabulary Count: {len(unique_vocab)}\n")

# if __name__ == "__main__":
#     from vllm import LLM, SamplingParams

#     llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
#     curriculum = StoryCurriculum(llm)
#     curriculum.generate_stories()
