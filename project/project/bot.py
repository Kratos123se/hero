pip install aiml





import aiml
import time
time.clock=time.time
# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("LOAD A")
# Press CTRL-C to break this loop
while True:
  message = input("Enter your message to the bot: ")
  if message == "quit":
    break
  else:
    bot_response = kernel.respond(message)
    print(bot_response)