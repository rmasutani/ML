import numpy as np
import heapq


class Node:
    def __init__(self, freq, num):
        self.left = None
        self.right = None
        self.freq = freq
        self.num = num

    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        return self.freq == other.freq

    def __gt__(self, other):
        return self.freq > other.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.code = {}

    def count_freq_dict(self, nums):
        # Count frequency and return as a dict
        out = dict()
        for i in nums:
            if i in out.keys():
                out[i] += 1
            else:
                out[i] = 1

        return out

    def create_heap(self, freq):
        # Function to store frequencies to a min heap
        for k, v in freq.items():
            node = Node(num=k, freq=v)
            heapq.heappush(self.heap, node)

    def merge_node(self, node1, node2):
        # Function to merge two smallest nodes and return a new node
        combined_node = Node(num=None, freq=node1.freq+node2.freq)
        combined_node.left = node1
        combined_node.right = node2

        return combined_node

    def build_huffman_tree(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = self.merge_node(node1, node2)
            heapq.heappush(self.heap, merged)

        tree = heapq.heappop(self.heap)

        return tree

    def create_code_dict(self, root, path):
        # Base case
        if not root:
            return

        if root.num != None:
            self.code[root.num] = path
            return

        self.create_code_dict(root.left, path+"0")
        self.create_code_dict(root.right, path+"1")

    def encode(self, nums):
        # Count frequency
        freq = self.count_freq_dict(nums)
        self.create_heap(freq)

        # Code table
        new = self.build_huffman_tree()
        self.create_code_dict(new, "")

        # Encode original nums
        out = ""
        for i in nums:
            out += self.code[i]

        return out

    def decode(self, s):
        pass


def main():
    np.random.seed(1)
    huff = HuffmanCoding()
    nums = np.random.randint(0, 5, size=30)

    ans = huff.encode(nums)
    print(ans)
    print(len(ans))


if __name__ == '__main__':
    main()