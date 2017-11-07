import sqlite3
import argparse
from pprint import pprint
import traceback
import sys

DATABASE_FILE = 'database.sqlite3'
DEFAULT_QUERY = 'select * from titles order by crawled_at desc'
FORBIDDEN_KEYWORDS = [ 'drop', 'truncate']
COMMIT_KEYWORDS = [ 'insert', 'update', 'delete' ]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Explore sqlite3 database file')
    parser.add_argument('-d', '--database', help='sqlite3 database file', default=DATABASE_FILE)
    parser.add_argument('-q', '--query', default=DEFAULT_QUERY, help='query to execute')
    args = parser.parse_args()
    
    # check for forbidden keywords in query
    forbiddens = [ keyword for keyword in FORBIDDEN_KEYWORDS if keyword in args.query.lower().split() ]
    if forbiddens:
        print('Forbidden keyword(s) found in query: {}'.format(forbiddens))
        sys.exit()

    # run query
    db = sqlite3.connect(args.database)
    try:
        cursor = db.cursor()
        cursor.execute(args.query)
        # need commit?
        commits = [ keyword for keyword in COMMIT_KEYWORDS if keyword in args.query.lower().split() ]
        if commits:            
            db.commit()
            print('success')
        else:
            pprint(cursor.fetchall())
    except Exception as ex:
        print(traceback.format_exc())
    finally:
        db.close()
