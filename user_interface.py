import wx
import wx.adv
import csv


# dummy function to test form capabilities
def ds_search(form_data):
    file = open('datasets/train.csv', mode='r', encoding='windows-1251')
    csv.register_dialect('dl', 'excel', delimiter=';')
    csv_file = csv.reader(file, 'dl')

    power = 0

    for line in csv_file:
        tpl = (line[1], line[2], line[3], line[4])
        if tpl == form_data:
            print(line)
            power = line[8]

    return power


class FromFrm(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title="Form")

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        num_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.date_ctrl = wx.adv.DatePickerCtrl(panel)
        self.time_ctrl = wx.adv.TimePickerCtrl(panel)

        self.cl_low = wx.SpinCtrlDouble(panel,
                                        min=0, max=100, style=wx.SP_ARROW_KEYS)
        self.cl_mid = wx.SpinCtrlDouble(panel,
                                        min=0, max=100, style=wx.SP_ARROW_KEYS)
        self.cl_high = wx.SpinCtrlDouble(panel,
                                         min=0, max=100, style=wx.SP_ARROW_KEYS)

        self.trm = wx.SpinCtrlDouble(panel, style=wx.SP_ARROW_KEYS, min=0, max=1, inc=0.1)

        btn = wx.Button(panel, label="Get Power")

        num_sizer.Add(self.cl_low, 0, wx.ALL | wx.EXPAND, 5)
        num_sizer.Add(self.cl_mid, 0, wx.ALL | wx.EXPAND, 5)
        num_sizer.Add(self.cl_high, 0, wx.ALL | wx.EXPAND, 5)

        sizer.Add(self.date_ctrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.time_ctrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(num_sizer, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.trm, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(btn, 0, wx.ALL | wx.CENTER, 5)

        panel.SetSizer(sizer)
        btn.Bind(wx.EVT_BUTTON, self.on_press)

    def on_press(self, event):
        date = self.date_ctrl.GetValue().Format("%d.%m.%Y")
        time = self.time_ctrl.GetValue().Format("%-H:%M")
        cl = (int(self.cl_low.GetValue()), int(self.cl_mid.GetValue()), int(self.cl_high.GetValue()))

        data = (f'{date} {time}', str(cl[0]), str(cl[1]), str(cl[2]))
        power = ds_search(data)

        msg = wx.MessageDialog(self, message=f'The power is {power}')
        msg.ShowModal()


if __name__ == '__main__':
    app = wx.App()
    frm = FromFrm()
    frm.Show()
    app.MainLoop()
