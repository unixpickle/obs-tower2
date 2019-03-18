import itertools
import time

from bs4 import BeautifulSoup
import requests

BASE_URL = 'https://www.aicrowd.com/challenges/unity-obstacle-tower-challenge/leaderboards'


def main():
    board = fetch_leaderboard()
    while True:
        time.sleep(60 * 5)
        new_board = fetch_leaderboard()
        print_diffs(board, new_board)
        board = new_board


def print_diffs(old, new):
    for k, v in new.items():
        if k not in old:
            print('new submission: %s -> %f' % (k, v))
        elif old[k] != v:
            print('new submission: %s -> %f (old: %f)' % (k, v, old[k]))
    for k, v in old.items():
        if k not in new:
            print('deleted submission: %s -> %f' % (k, v))


def fetch_leaderboard():
    result = {}
    for page in itertools.count():
        sub_result = fetch_leaderboard_page(page)
        if not len(sub_result):
            break
        result.update(sub_result)
    return result


def fetch_leaderboard_page(page):
    url = '%s?page=%d' % (BASE_URL, page)
    response = requests.get(url)
    data = response.text
    soup = BeautifulSoup(data, 'html.parser')
    table = soup.find('table', {'class': 'table-leaderboard'})
    result = {}
    for row in table.find('tbody').find_all('tr'):
        columns = row.find_all('td')
        result[columns[2].get_text().strip()] = float(columns[4].get_text().strip())
    return result


if __name__ == '__main__':
    main()
