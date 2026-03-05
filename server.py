from openreward.environments import Server

from open_rl import OpenRL

if __name__ == "__main__":
    server = Server([OpenRL])
    server.run()
