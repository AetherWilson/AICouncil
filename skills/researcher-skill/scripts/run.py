import json
import sys


def main():
    payload = {
        'tool': 'researcher-skill/run.py',
        'args': sys.argv[1:],
        'note': 'Default researcher tool entrypoint. Replace with project-specific logic as needed.'
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
