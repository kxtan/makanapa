<script lang="ts">
  import ChatInterface from '$lib/ChatInterface.svelte';
  import { userContext } from '$lib/stores';
  
  // Initialize user context
  $userContext = {
    location: null,
    meal_type: null,
    dietary_restrictions: null,
    group_size: null,
    budget: null,
    mood: null,
    cuisine_preference: null,
    timezone: null,
    local_time: null,
    coords: null,
    location_label: null,
  };

  // Auto-detect timezone and local time
  let detectTimezoneAndLocalTime = () => {
    try {
      const tzName = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const nowLocal = new Date();
      
      if (tzName) {
        $userContext.timezone = tzName;
      }
      
      $userContext.local_time = nowLocal.toISOString();
      
      // Infer meal type
      const h = nowLocal.getHours();
      if (5 <= h && h < 11) $userContext.meal_type = "breakfast";
      else if (11 <= h && h < 15) $userContext.meal_type = "lunch";
      else if (15 <= h && h < 18) $userContext.meal_type = "tea/snack";
      else if (18 <= h && h < 22) $userContext.meal_type = "dinner";
      else $userContext.meal_type = "supper";
      
    } catch (e) {
      console.error("Error detecting timezone:", e);
    }
  };

  // Auto-detect geolocation
  let detectGeolocation = async () => {
    if (!navigator.geolocation) return;
    
    try {
      const position = await new Promise<GeolocationPosition>((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000
        });
      });
      
      $userContext.coords = {
        lat: position.coords.latitude,
        lon: position.coords.longitude,
        accuracy: position.coords.accuracy
      };
      
      // Simple reverse geocoding
      $userContext.location = "Your current location";
      $userContext.location_label = `Lat: ${position.coords.latitude.toFixed(4)}, Lon: ${position.coords.longitude.toFixed(4)}`;
      
    } catch (e) {
      console.error("Geolocation error:", e);
    }
  };

  // Initialize on component mount
  import { onMount } from 'svelte';
  
  onMount(() => {
    detectTimezoneAndLocalTime();
    detectGeolocation();
  });
</script>

<div class="app-container">
  <header class="app-header">
    <h1>🍜 MakanApa</h1>
    <p class="subtitle">Your friendly food discovery companion - Let's figure out what to eat!</p>
  </header>
  
  <main class="main-content">
    <ChatInterface />
  </main>
</div>

<style>
  .app-container {
    min-height: 100vh;
    background: #0f172a;
    color: #e2e8f0;
  }

  .app-header {
    background: #1e293b;
    padding: 1.5rem 2rem;
    text-align: center;
    border-bottom: 1px solid #334155;
  }

  .app-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 0.5rem 0;
  }

  .subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 500;
  }

  .main-content {
    padding: 0;
  }

  @media (max-width: 768px) {
    .app-header {
      padding: 1rem;
    }
    
    .app-header h1 {
      font-size: 2rem;
    }
    
    .subtitle {
      font-size: 1rem;
    }
  }
</style>
