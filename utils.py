def prompt(text: str, actions: dict[str, callable]) -> None:
    """
    :param text: Tekst zachęty, który zostanie wyświetlony przed wyborem użytkownika.
    :param actions: Słownik akcji gdzie kluczem jest odpowiedź użytkownika a wartością funkcja, która ma zostać wykonana.
    :return: `None`
    """
    while True:
        resp = input(text)
        if resp in actions:
            actions[resp]()
            break

class ParkingSpace:
    def __init__(self, pos, width, height):
        self.pos = pos
        self.width = width
        self.height = height

    def inMe(self, x, y):
        x1, y1 = self.pos
        x2, y2 = x1 + self.width, y1 + self.height
        return (x1 < x < x2) and (y1 < y < y2)

    def beginEnd(self):
        return self.pos, (self.pos[0] + self.width, self.pos[1] + self.height)