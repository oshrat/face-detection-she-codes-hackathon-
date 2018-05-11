

import tkinter
import os
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

class simpleapp_tk(tkinter.Tk):
    def __init__(self,parent):
        tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()

        self.entryVariable = tkinter.StringVar()
        self.entry = tkinter.Entry(self,textvariable=self.entryVariable)
        self.entry.grid(column=0,row=0,sticky='EW')
        self.entry.bind("<Return>", self.OnPressEnter)
        self.entryVariable.set(u"Enter text here.")

        button = tkinter.Button(self,text=u"Click me !",
                                command=self.OnButtonClick)
        button.grid(column=1,row=0)

        self.labelVariable = tkinter.StringVar()
        label = tkinter.Label(self,textvariable=self.labelVariable,
                              anchor="w",fg="white",bg="blue")
        label.grid(column=0,row=1,columnspan=2,sticky='EW')
        self.labelVariable.set(u"Hello !")

        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,False)
        self.update()
        self.geometry(self.geometry())
        self.entry.focus_set()
        self.entry.selection_range(0, self.tkinter.END)

    def OnButtonClick(self):
        self.labelVariable.set( self.entryVariable.get())
        self.entry.focus_set()
        self.entry.selection_range(0, tkinter.END)

    def OnPressEnter(self,event):
        self.labelVariable.set( self.entryVariable.get())
        self.entry.focus_set()
        self.entry.selection_range(0, tkinter.END)

class Player(object):
    def __init__(self):
        self.root = tkinter.Tk()
        self.w, self.h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (self.w, self.h))
        Gst.init(None)
        GObject.threads_init()

        self.video = tkinter.Frame(self.root, bg='#000000')
        self.video.grid(row=0, column=0, sticky="nsew")

        self.frame_id = self.video.winfo_id()

        self.playbin = Gst.ElementFactory.make('self.playbin', None)
        self.playbin.set_property('video-sink', None)
        self.playbin.set_property('uri', 'file://%s' % (os.path.abspath('out.mpeg')))
        self.playbin.set_state(Gst.State.PLAYING)

        bus = self.playbin.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect('sync-message::element', self.set_frame_handle, self.frame_id)
        self.root.mainloop()
if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('my application')

    app.mainloop()
