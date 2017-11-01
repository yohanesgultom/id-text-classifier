import requests
import json
import ConfigParser
import sqlite3
import argparse

CONFIG_FILE='.config'

config = ConfigParser.ConfigParser()
config.readfp(open(CONFIG_FILE))

def get_page_info(page_id, page_token):
  params = {
    'fields': 'about,attire,bio,location,parking,hours,emails,website',
    'access_token': page_token
  }
  r = requests.get(config.get('FACEBOOK', 'API_URL') + '/' + page_id, params=params)
  return r.json()


def post_to_page(page_id, page_token, message, link):
  params = {
    'message': message,
    'link': link,
    'access_token': page_token
  }
  r = requests.post(config.get('FACEBOOK', 'API_URL') + '/' + page_id + '/feed', data=params)
  return r.json()


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Post a title from database to facebook page')
  parser.add_argument('-s', '--silent', action='store_true', default=False, help='display no log in console')
  args = parser.parse_args()
  verbose = not args.silent

  # load db
  db = sqlite3.connect(config.get('APP', 'SQLITE3_DB'))
  cursor = db.cursor()

  # fetch one oldest record
  cursor.execute('''SELECT * FROM titles WHERE posted_at IS NULL ORDER BY crawled_at ASC LIMIT 1''')
  record = cursor.fetchone()

  if record:
    # post to fb page
    id, title, link, crawled_at, posted_at, post_id = record
    try:
      resp = post_to_page(
        config.get('FACEBOOK', 'PAGE_ID'), 
        config.get('FACEBOOK', 'PAGE_TOKEN'),
        None,  
        link
      )

      if verbose:
        print(resp)
      # update database
      cursor.execute('''UPDATE titles SET post_id = ?, posted_at = datetime('now', 'localtime') WHERE id = ?''', (resp['id'], id, ))
      if verbose:
        print('Database updated')
      db.commit()

    except Exception as e:
      print('Error: {}'.format(e))
      
  else:
    if verbose:
      print('No new link available in database. Please crawl first')