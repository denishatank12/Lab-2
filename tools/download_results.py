"""Offer one ZIP through the notebook channel, including Colab with a local kernel.

Uses the public Colab kernel-comms download protocol demonstrated by Google's
Apache-2.0 colabtools/google/colab/files.py. No HTTP server or credentials needed.
"""
from pathlib import Path
import json
import os
import platform
import shutil
import uuid


def register_zip_transfer(shell,path):
    """Serve only this file in bounded chunks; never expose a directory."""
    path=Path(path).resolve()
    if not path.is_file() or path.suffix.lower()!='.zip':
        raise FileNotFoundError('Create a complete result ZIP first.')
    target='lab2_zip_'+uuid.uuid4().hex
    manager=shell.kernel.comm_manager
    def opened(comm,message):
        stream=path.open('rb')
        def close(message=None):
            stream.close()
        def read(message):
            if stream.closed:
                return
            try:
                block=stream.read(1024*1024)
                if block:
                    comm.send({'bytes':len(block)},buffers=[block])
                else:
                    stream.close()
                    comm.close()
            except Exception:
                stream.close()
                comm.close()
                raise
        comm.on_msg(read)
        comm.on_close(close)
    manager.register_target(target,opened)
    return target


def offer_download(path,auto_start=True):
    path=Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError('The result ZIP does not exist yet.')
    # A native PC kernel can copy directly. A Docker /root directory is not Windows Downloads.
    destination=os.environ.get('LAB2_DOWNLOAD_DIR')
    native_downloads=Path.home()/'Downloads'
    container=Path('/.dockerenv').exists() or Path('/run/.containerenv').exists()
    if destination or (not container and platform.system() in {'Windows','Darwin'} and native_downloads.is_dir()):
        folder=Path(destination).expanduser() if destination else native_downloads
        if not folder.is_dir():
            raise FileNotFoundError('LAB2_DOWNLOAD_DIR must point to an existing runtime-accessible folder.')
        copied=folder/path.name
        if copied.resolve()!=path:
            if copied.exists():
                raise FileExistsError('A same-named download already exists; it was not overwritten.')
            shutil.copy2(path,copied)
        print('ZIP copied to Downloads on the runtime machine:',copied)
    from IPython import get_ipython
    from IPython.display import Javascript,display
    shell=get_ipython()
    if shell is None or not hasattr(shell,'kernel'):
        print('ZIP:',path)
        print('Run the notebook download cell to transfer it through Colab, or copy this file from the runtime machine.')
        return
    target=register_zip_transfer(shell,path)
    # Keep a clickable Blob link after transfer, in case the browser blocks the automatic save.
    script=r'''(function(cfg) {
      const panel=document.createElement('div');
      const button=document.createElement('button');
      button.textContent='Download all Lab 2 results';
      button.style.cssText='padding:10px 16px;margin:8px 0;cursor:pointer';
      const status=document.createElement('div');
      status.textContent='ZIP is ready. The browser uses your normal download settings.';
      panel.append(button,status);document.body.appendChild(panel);
      async function transfer() {
        if (!globalThis.google?.colab?.kernel?.comms) {
          status.textContent='Not a Colab frontend. Use the runtime Downloads copy, or copy the printed ZIP path. No localhost link is required.';
          return;
        }
        if (!google.colab.kernel.accessAllowed) {
          status.textContent='Allow this notebook output to access the kernel, then click Download.';
          return;
        }
        button.disabled=true;
        const chunks=[];let received=0;let channel;
        try {
          channel=await google.colab.kernel.comms.open(cfg.target);
          channel.send({read:true});
          for await (const packet of channel.messages) {
            if (packet.buffers) for (const chunk of packet.buffers) {
              chunks.push(chunk);received+=chunk.byteLength;
              status.textContent=`Transferring ${(received/1048576).toFixed(1)} / ${(cfg.bytes/1048576).toFixed(1)} MiB`;
            }
            channel.send({read:true});
          }
          if (received!==cfg.bytes) throw Error('Transfer incomplete. Keep the runtime connected and retry this download cell.');
          const url=URL.createObjectURL(new Blob(chunks,{type:'application/zip'}));
          const save=document.createElement('a');save.href=url;save.download=cfg.name;
          save.textContent='Save ZIP again if your browser blocked the automatic download';
          panel.appendChild(save);save.click();
          status.textContent='Transfer finished. Check your browser downloads, then upload the ZIP in this chat. No training was repeated.';
          button.textContent='Transfer ZIP again';
        } catch(error) {
          status.textContent=String(error)+' The ZIP remains saved on the runtime.';
        } finally {button.disabled=false;}
      }
      button.onclick=transfer;
      if (cfg.autoStart) transfer();
    })(CONFIG);'''
    script=script.replace('CONFIG',json.dumps(dict(target=target,name=path.name,bytes=path.stat().st_size,autoStart=auto_start)))
    display(Javascript(script))
    print('ZIP remains at:',path)
    print('The notebook kernel must remain connected during transfer. Browser download location follows your settings.')
