try:
    from lexos.io.smart import Loader
except ImportError:
    print("Failed to import Loader.")

texts = ["This is a test.", "This is another test."]

loader = Loader()
loader.texts = texts

for text in loader.texts:
    print(text)

print("Done!")
