(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const LOCAL_PORTAL = ['localhost', '127.0.0.1'].includes(window.location.hostname) && window.location.port === '8765';
  const DESKTOP_BRIDGE_URL = LOCAL_PORTAL ? '' : 'http://127.0.0.1:8765';
  const splash = $('splash');
  const authPage = $('authPage');
  const workspace = $('workspace');
  const ownerView = $('ownerView');
  const loginForm = $('loginForm');
  const registerForm = $('registerForm');
  const authMessage = $('authMessage');
  const toast = $('toast');
  let toastTimer = null;
  let currentUser = null;
  let adminTableMode = 'users';
  let statusPoll = null;

  document.querySelectorAll('[data-year]').forEach(node => node.textContent = String(new Date().getFullYear()));

  function showToast(message, kind = 'info') {
    toast.textContent = message;
    toast.dataset.kind = kind;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 4800);
  }

  function authNotice(message, success = false) {
    authMessage.textContent = message;
    authMessage.classList.toggle('success', Boolean(success));
  }

  async function api(path, options = {}) {
    const init = {
      method: options.method || 'GET',
      credentials: 'same-origin',
      headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) },
      ...(options.body ? { body: JSON.stringify(options.body) } : {})
    };
    let response;
    try {
      response = await fetch(path, init);
    } catch (error) {
      throw new Error('Cannot reach the AIRWRITE cloud service. Please check your internet connection and try again.');
    }
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json') ? await response.json() : null;
    if (!response.ok || (payload && payload.ok === false)) {
      throw new Error(payload?.error || `Request failed (${response.status}).`);
    }
    return payload || {};
  }

  // Account data uses the hosted API. Camera and desktop controls always use
  // the user's own Windows bridge; the hosted server cannot launch a remote PC app.
  async function desktopApi(path, options = {}) {
    if (!DESKTOP_BRIDGE_URL) return api(path, options);
    let response;
    try {
      response = await fetch(`${DESKTOP_BRIDGE_URL}${path}`, {
        method: options.method || 'GET',
        mode: 'cors',
        credentials: 'omit',
        headers: options.body ? { 'Content-Type': 'application/json' } : {},
        ...(options.body ? { body: JSON.stringify(options.body) } : {})
      });
    } catch (_) {
      throw new Error('Desktop bridge not found. On this Windows PC, run START_AIRWRITE.bat and allow local-network access in your browser.');
    }
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json') ? await response.json() : null;
    if (!response.ok || (payload && payload.ok === false)) {
      throw new Error(payload?.error || `Desktop bridge request failed (${response.status}).`);
    }
    return payload || {};
  }

  function showAuth(mode = 'login') {
    splash.classList.add('leaving');
    authPage.classList.remove('hidden');
    workspace.classList.add('hidden');
    ownerView.classList.add('hidden');
    setAuthMode(mode);
  }

  function showWorkspace(user) {
    currentUser = user;
    splash.classList.add('leaving');
    authPage.classList.add('hidden');
    ownerView.classList.add('hidden');
    workspace.classList.remove('hidden');
    const name = user.full_name || 'AIRWRITE user';
    $('accountName').textContent = name;
    $('accountAvatar').textContent = name.trim().charAt(0).toUpperCase() || 'A';
    $('welcomeLine').textContent = `WELCOME BACK, ${name.split(/\s+/)[0].toUpperCase()}`;
    $('ownerNav').classList.toggle('hidden', user.role !== 'admin');
    if (statusPoll) clearInterval(statusPoll);
    refreshStatus();
    statusPoll = setInterval(refreshStatus, 4500);
  }

  function setAuthMode(mode) {
    const register = mode === 'register';
    loginForm.classList.toggle('hidden', register);
    registerForm.classList.toggle('hidden', !register);
    $('loginTab').classList.toggle('active', !register);
    $('registerTab').classList.toggle('active', register);
    $('loginTab').setAttribute('aria-selected', String(!register));
    $('registerTab').setAttribute('aria-selected', String(register));
    $('authTitle').textContent = register ? 'Create your account.' : 'Welcome back.';
    $('authSubtitle').textContent = register
      ? 'Create an AIRWRITE account and securely save it to the cloud.'
      : 'Sign in to launch your AIRWRITE desktop experience.';
    authNotice('');
  }

  $('enterPortal').addEventListener('click', () => showAuth('login'));
  setTimeout(() => {
    if (!splash.classList.contains('leaving')) showAuth('login');
  }, 4200);
  $('loginTab').addEventListener('click', () => setAuthMode('login'));
  $('registerTab').addEventListener('click', () => setAuthMode('register'));

  document.querySelectorAll('[data-toggle-password]').forEach(button => {
    button.addEventListener('click', () => {
      const input = $(button.dataset.togglePassword);
      if (!input) return;
      input.type = input.type === 'password' ? 'text' : 'password';
      button.textContent = input.type === 'password' ? 'Show' : 'Hide';
    });
  });

  loginForm.addEventListener('submit', async event => {
    event.preventDefault();
    if (!loginForm.reportValidity()) return;
    const submit = loginForm.querySelector('button[type="submit"]');
    submit.disabled = true;
    authNotice('Verifying your account…');
    try {
      const result = await api('/api/login', { method: 'POST', body: {
        identifier: $('loginIdentifier').value.trim(),
        password: $('loginPassword').value
      }});
      loginForm.reset();
      showWorkspace(result.user);
      showToast(`Welcome to AIRWRITE, ${result.user.full_name}.`, 'success');
    } catch (error) {
      authNotice(error.message);
    } finally {
      submit.disabled = false;
    }
  });

  registerForm.addEventListener('submit', async event => {
    event.preventDefault();
    if (!registerForm.reportValidity()) return;
    if ($('registerPassword').value.length < 8) {
      authNotice('Use a password with at least 8 characters.'); return;
    }
    const submit = registerForm.querySelector('button[type="submit"]');
    submit.disabled = true;
    authNotice('Creating your secure cloud account…');
    try {
      const result = await api('/api/register', { method: 'POST', body: {
        full_name: $('registerName').value.trim(),
        organisation: $('registerOrg').value.trim(),
        email: $('registerEmail').value.trim(),
        password: $('registerPassword').value,
        account_type: 'Personal',
        consent: $('registerConsent').checked
      }});
      registerForm.reset();
      showWorkspace(result.user);
      showToast('Your AIRWRITE account was created securely.', 'success');
    } catch (error) {
      authNotice(error.message);
    } finally {
      submit.disabled = false;
    }
  });

  async function refreshStatus() {
    try {
      const data = await desktopApi('/api/status');
      $('bridgeState').textContent = 'Connected';
      $('portalStatus').textContent = 'Connected';
      $('cameraProfileStatus').textContent = data.camera_profile_present ? 'Saved' : 'Not found';
      $('calibrationStatus').textContent = `${data.calibration_profile_points || 0} reference points`;
      const running = Boolean(data.desktop_running);
      $('desktopState').textContent = running ? 'DESKTOP ACTIVE' : 'DESKTOP IDLE';
      $('engineStatus').textContent = running ? 'Running' : 'Ready to launch';
      $('engineDot').className = `status-indicator ${running ? 'green' : 'amber'}`;
      ['launchLaser', 'launchHand'].forEach(id => {
        const button = $(id);
        if (button) button.disabled = running;
      });
      if (running) {
        const activeName = data.desktop_mode === 'hand' ? 'Hand Gesture Studio' : 'Laser Writer';
        $('engineStatus').textContent = `${activeName} active`;
        $('bridgeState').textContent = `${activeName} running`;
      }
    } catch (_) {
      $('bridgeState').textContent = 'Offline';
      $('portalStatus').textContent = 'Not connected';
      ['launchLaser', 'launchHand'].forEach(id => { if ($(id)) $(id).disabled = true; });
    }
  }

  async function startDesktopMode(mode, buttonId) {
    const button = $(buttonId);
    if (!button || button.disabled) return;
    const other = $(buttonId === 'launchLaser' ? 'launchHand' : 'launchLaser');
    button.disabled = true;
    if (other) other.disabled = true;
    const copy = button.querySelector('.launch-mode-copy small');
    if (copy) copy.textContent = mode === 'hand' ? 'Opening the selected webcam and hand-tracking engine…' : 'Opening the laser camera and calibration workflow…';
    try {
      const result = await desktopApi('/api/desktop/start', { method: 'POST', body: { mode } });
      showToast(result.message || 'AIRWRITE desktop launch requested.', 'success');
      setTimeout(refreshStatus, 1000);
      setTimeout(refreshStatus, 2600);
    } catch (error) {
      showToast(error.message, 'error');
      button.disabled = false;
      if (other) other.disabled = false;
      if (copy) copy.textContent = mode === 'hand'
        ? 'Open the laptop webcam · learn gestures · draw, pan, zoom and control tools; no hand calibration'
        : 'Open the real desktop app · camera preview · quick 5-point calibration · Smart Board';
    }
  }

  $('launchLaser').addEventListener('click', () => startDesktopMode('laser', 'launchLaser'));
  $('launchHand').addEventListener('click', () => startDesktopMode('hand', 'launchHand'));

  $('logoutButton').addEventListener('click', async () => {
    try { await api('/api/logout', { method: 'POST' }); } catch (_) { /* Cloud logout continues locally in the browser. */ }
    currentUser = null;
    if (statusPoll) clearInterval(statusPoll);
    showAuth('login');
    showToast('You have signed out.');
  });

  $('ownerNav').addEventListener('click', openOwnerView);
  $('backToWorkspace').addEventListener('click', () => {
    ownerView.classList.add('hidden');
    workspace.classList.remove('hidden');
  });
  $('usersTab').addEventListener('click', () => { adminTableMode = 'users'; loadOwnerData(); });
  $('enquiriesTab').addEventListener('click', () => { adminTableMode = 'enquiries'; loadOwnerData(); });
  $('refreshAdmin').addEventListener('click', loadOwnerData);

  async function openOwnerView() {
    if (currentUser?.role !== 'admin') return;
    workspace.classList.add('hidden');
    ownerView.classList.remove('hidden');
    await loadOwnerData();
  }

  function cell(row, value) {
    const td = document.createElement('td');
    td.textContent = value == null || value === '' ? '—' : String(value);
    row.appendChild(td);
  }

  async function loadOwnerData() {
    if (currentUser?.role !== 'admin') return;
    const head = $('adminHead');
    const body = $('adminBody');
    head.replaceChildren(); body.replaceChildren();
    try {
      const [users, enquiries] = await Promise.all([
        api('/api/admin/users'), api('/api/admin/enquiries')
      ]);
      $('registeredCount').textContent = String(users.count);
      $('enquiryCount').textContent = String(enquiries.count);
      const headings = adminTableMode === 'users'
        ? ['NAME', 'EMAIL', 'ORGANISATION', 'REGISTERED', 'LAST LOGIN', 'LOGINS']
        : ['NAME', 'EMAIL', 'ORGANISATION', 'ROLE', 'CITY', 'INTEREST', 'SUBMITTED'];
      const headerRow = document.createElement('tr');
      headings.forEach(label => { const th = document.createElement('th'); th.textContent = label; headerRow.appendChild(th); });
      head.appendChild(headerRow);
      const rows = adminTableMode === 'users' ? users.users : enquiries.enquiries;
      if (!rows.length) {
        const tr = document.createElement('tr'); const td = document.createElement('td');
        td.colSpan = headings.length; td.textContent = adminTableMode === 'users' ? 'No user registrations yet.' : 'No pilot enquiries yet.';
        tr.appendChild(td); body.appendChild(tr); return;
      }
      rows.forEach(item => {
        const tr = document.createElement('tr');
        if (adminTableMode === 'users') {
          [item.full_name, item.email, item.organisation, formatDate(item.created_at), formatDate(item.last_login), item.login_count].forEach(v => cell(tr, v));
        } else {
          [item.name, item.email, item.organisation, item.role, item.city, item.interest, formatDate(item.created_at)].forEach(v => cell(tr, v));
        }
        body.appendChild(tr);
      });
    } catch (error) {
      const tr = document.createElement('tr'); const td = document.createElement('td');
      td.colSpan = 7; td.textContent = error.message; tr.appendChild(td); body.appendChild(tr);
    }
    $('usersTab').classList.toggle('active', adminTableMode === 'users');
    $('enquiriesTab').classList.toggle('active', adminTableMode === 'enquiries');
  }

  function formatDate(value) {
    if (!value) return 'Never';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
  }

  $('pilotForm').addEventListener('submit', async event => {
    event.preventDefault();
    const form = $('pilotForm');
    if (!form.reportValidity()) return;
    const button = form.querySelector('button[type="submit"]');
    button.disabled = true;
    $('pilotMessageStatus').textContent = 'Saving your enquiry securely…';
    const values = Object.fromEntries(new FormData(form).entries());
    try {
      const result = await api('/api/pilot', { method: 'POST', body: values });
      form.reset();
      $('pilotCountry').value = 'India';
      $('pilotMessageStatus').textContent = result.message;
      showToast('Pilot enquiry saved to the AIRWRITE owner account.', 'success');
    } catch (error) {
      $('pilotMessageStatus').textContent = error.message;
      showToast(error.message, 'error');
    } finally {
      button.disabled = false;
    }
  });

  async function restoreSession() {
    try {
      const data = await api('/api/me');
      if (data.user) showWorkspace(data.user);
      else showAuth('login');
    } catch (_) {
      showAuth('login');
    }
  }

  // Check existing session while the short intro is playing. If not signed in,
  // the splash transitions to the authentication screen after its animation.
  restoreSession();
})();
