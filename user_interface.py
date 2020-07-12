import wx
import wx.adv
import csv


# dummy function to test form capabilities
def get_points(obj_id):
    pass


def ds_search(form_data):
    file = open('datasets/train.csv', mode='r', encoding='windows-1251')
    csv.register_dialect('dl', 'excel', delimiter=';')
    csv_file = csv.reader(file, 'dl')

    power = 0

    id_set = set()

    for line in csv_file:
        if line[0] not in id_set:
            id_set.add(line[0])
        tpl = (line[0], line[1], line[2], line[3], line[4])
        if tpl == form_data:
            power = line[8]

    print(id_set)

    return power


class InputPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent=parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        cl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        obj_ids = ['22880', '50606', '61928', '95631', '87390', '21645', '53373', '53759', '95646', '33894',
                   '20238', '61577', '54207', '44227', '89943', '54284', '25092', '55637', '35872', '40927',
                   '54699', '45827', '18512', '56392', '19829', '43017', '88297', '59875', '11360', '49076',
                   '97406', '99086', '48883', '29217', '66345', '90422', '96220', '88648', '61518', '97908',
                   '10769', '93422', '68359', '90497']

        self.obj_choice = wx.Choice(self, choices=obj_ids)
        self.obj_choice.SetSelection(0)
        self.date_ctrl = wx.adv.DatePickerCtrl(self)
        self.time_ctrl = wx.adv.TimePickerCtrl(self)

        self.cl_low = wx.SpinCtrlDouble(self,
                                        min=0, max=100, style=wx.SP_ARROW_KEYS)
        self.cl_mid = wx.SpinCtrlDouble(self,
                                        min=0, max=100, style=wx.SP_ARROW_KEYS)
        self.cl_high = wx.SpinCtrlDouble(self,
                                         min=0, max=100, style=wx.SP_ARROW_KEYS)

        self.trm = wx.SpinCtrlDouble(self, style=wx.SP_ARROW_KEYS, min=0, max=1, inc=0.1)

        self.btn = wx.Button(self, label="Get Power")

        cl_sizer.Add(self.cl_low, 0, wx.ALL | wx.EXPAND, 5)
        cl_sizer.Add(self.cl_mid, 0, wx.ALL | wx.EXPAND, 5)
        cl_sizer.Add(self.cl_high, 0, wx.ALL | wx.EXPAND, 5)

        sizer.Add(self.obj_choice, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.date_ctrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.time_ctrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(cl_sizer, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.trm, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.btn, 0, wx.ALL | wx.CENTER, 5)

        self.SetSizer(sizer)


class FromFrm(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title="Form")

        self.panel = InputPanel(self)
        self.panel.btn.Bind(wx.EVT_BUTTON, self.on_press)

    def on_press(self, event):
        obj = self.panel.obj_choice.GetStringSelection()
        date = self.panel.date_ctrl.GetValue().Format("%d.%m.%Y")
        time = self.panel.time_ctrl.GetValue().Format("%-H:%M")
        cl = (int(self.panel.cl_low.GetValue()),
              int(self.panel.cl_mid.GetValue()), int(self.panel.cl_high.GetValue()))

        data = (obj, f'{date} {time}', str(cl[0]), str(cl[1]), str(cl[2]))
        power = ds_search(data)

        msg = wx.MessageDialog(self, message=f'The power is {power}')
        msg.ShowModal()


if __name__ == '__main__':
    app = wx.App()
    frm = FromFrm()
    frm.Show()
    app.MainLoop()
