from tkinter import ttk

class CmbxSettings():
    
    def __init__(self, name: str, label: str, values: list, text: list):
        self.name = name
        self.label = label
        text = [' '] + text
        values = [None] + values
        self.map: dict[str: any] = {text[i]: values[i] for i in range(len(values))}

class CmbxBuilder():
    
    def __init__(self, settings: dict[CmbxSettings], parent: object):
        self.parent = parent
        self.settings: dict[CmbxSettings] = settings
        self.boxes = {}
        
    def build(self) -> None:
        """Builds the comboboxes."""
        cntr = 0
        for cmbx_settings in self.settings.values():
            self.boxes[cmbx_settings.name] = self.build_combobox(cmbx_settings, cntr)
            cntr += 1
            
    def build_combobox(self, cmbx_settings: CmbxSettings, index: int) -> ttk.Combobox:
        """Builds a combobox.
        
        Args:
            cmbx_settings (CmbxSettings): The settings for the combobox.
            index (int): The index of the combobox.
            
        Returns:
            ttk.Combobox: The combobox.
        """
        ttk.Label(self.parent, text=cmbx_settings.label).grid(row=index, column=0, sticky='w')
        box = ttk.Combobox(self.parent, values=[str(i) for i in cmbx_settings.map.keys()], state='readonly')
        box.grid(row=index, column=1, sticky='we')
        box.set(' ')
        return box
    
    def get_values(self) -> dict:
        """Gets the values of the comboboxes.
        
        Returns:
            dict: The values of the comboboxes.       
        """
        values = {}
        for name, box in self.boxes.items():
            text = box.get()
            if text != ' ':
                values[name] = [self.settings[name].map[text]]
        return values
    
    def set_states(self, state: str) -> None:
        """Sets the state of the comboboxes.
        
        Returns:
            dict: The values of the comboboxes.       
        """        
        for box in self.boxes.values():
            box['state'] = state