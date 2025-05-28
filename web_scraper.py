import requests
from bs4 import BeautifulSoup
from collections import deque, defaultdict
import time
import networkx as nx

def get_links_and_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        base = "/".join(url.split("/")[:3])

        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('http'):
                links.add(href)
            elif href.startswith('/'):
                links.add(base + href)

        text = soup.get_text()
        print(f"[+] Fetched: {url}, Links: {len(links)}, Length of text: {len(text)}")
        return list(links), text
    except Exception as e:
        print(f"[-] Failed: {url}, Error: {e}")
        return [], ""

def bfs(start_url, keyword, max_depth=2, visit_limit=20):
    visited = set()
    queue = deque([(start_url, 0)])
    graph = defaultdict(list)
    total_occurrences = 0
    pages_visited = 0
    start_time = time.time()

    while queue and pages_visited < visit_limit:
        current_url, depth = queue.popleft()
        if current_url in visited or depth > max_depth:
            continue
        visited.add(current_url)
        pages_visited += 1

        links, text = get_links_and_text(current_url)
        total_occurrences += text.lower().count(keyword.lower())

        for link in links:
            # Enqueue only if not visited and visit limit not exceeded
            if link not in visited and (len(queue) + pages_visited) < visit_limit:
                queue.append((link, depth + 1))
                graph[current_url].append(link)

    end_time = time.time()
    return graph, total_occurrences, pages_visited, end_time - start_time

def dfs(start_url, keyword, max_depth=2, visit_limit=20):
    visited = set()
    stack = [(start_url, 0)]
    graph = defaultdict(list)
    total_occurrences = 0
    pages_visited = 0
    start_time = time.time()

    while stack and pages_visited < visit_limit:
        current_url, depth = stack.pop()
        if current_url in visited or depth > max_depth:
            continue
        visited.add(current_url)
        pages_visited += 1

        links, text = get_links_and_text(current_url)
        total_occurrences += text.lower().count(keyword.lower())

        for link in links:
            if link not in visited and (len(stack) + pages_visited) < visit_limit:
                stack.append((link, depth + 1))
                graph[current_url].append(link)

    end_time = time.time()
    return graph, total_occurrences, pages_visited, end_time - start_time

def calculate_pagerank(graph):
    G = nx.DiGraph()
    for src, dests in graph.items():
        for dest in dests:
            G.add_edge(src, dest)
    return nx.pagerank(G)
