class PriceFormatter:
    
    def format(self, price: float) -> str:
        return f"$ {price * 100000:,.2f}"
